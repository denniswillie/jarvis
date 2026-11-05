"""Opt-in, single-event-loop snapshot publication; not a source authorization layer."""

import asyncio
import json
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID

from app.knowledge.models import CanonicalOwnerBusy, SourceError
from app.live_state.models import StateCursor
from app.live_state.task_context import TaskSnapshotControlChange, TaskSnapshotProgress
from app.voice_routes.contracts import ContextReceipt, ContextWriteEvidence, RouteIdentity, VoiceContextUpdate
from app.voice_routes.protocol import VoiceRoute


ContextValidity = Callable[[VoiceContextUpdate], Awaitable[bool]]
Clock = Callable[[], datetime]

_PUBLICATION_LOGGER = logging.getLogger(__name__)
_PUBLICATION_LOG_LIMIT = 128
_CANCELLATION_JOIN_LIMIT_SECONDS = 0.1
_PUBLICATION_STAGES = frozenset({
    "validity_entry", "validity_exit", "adapter_entry", "adapter_return",
    "publication_deadline", "publication_skipped", "cancellation_join_entry",
    "cancellation_join_return", "cancellation_join_deadline",
    "retained_context_validity_entry", "retained_context_validity_exit", "publication_deferred",
    "publication_fenced",
    "context_refresh_wait", "context_refresh_return",
    "mutation_budget_entry",
})
_PUBLICATION_OUTCOMES = frozenset({
    "started", "valid", "invalid", "returned", "cancelled", "failed",
    "superseded_preflight", "failed_closed", "terminal", "cleanup_unconfirmed",
    "proven_unwritten",
    "cancellation_pending",
    "expired_preflight", "expired_unwritten_adapter", "expired_after_acknowledged_delete", "fresh_snapshot_written",
    "unavailable_after_acknowledged_delete",
    "unavailable_in_acknowledged_empty_context",
    "expired_after_acknowledged_create",
    "deadline_after_acknowledged_delete",
    "task_progress_unwritten",
    "task_progress_after_acknowledged_delete",
    "task_control_unwritten", "task_control_after_acknowledged_delete",
    "owner_busy_unwritten", "owner_busy_after_acknowledged_delete",
})


@dataclass
class _PublicationProgress:
    started_at: float
    phase_started_at: float
    phase: str = "preflight"
    adapter_entered: bool = False
    write_evidence: ContextWriteEvidence = field(default_factory=ContextWriteEvidence)
    mutation_started_at: float | None = None
    normal_turn_pause: bool = False


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class OfferDisposition(str, Enum):
    QUEUED = "queued"
    COALESCED = "coalesced"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ContextOfferResult:
    disposition: OfferDisposition
    reason: str | None = None


@dataclass(frozen=True)
class ContextBridgeStatus:
    identity: RouteIdentity
    running: bool
    fenced: bool
    reason: str | None
    newest_cursor: StateCursor
    last_receipt: ContextReceipt | None
    retry_update_id: UUID | None = None
    refill_update_id: UUID | None = None


class _PublicationRejected(Exception):
    pass


class VoiceContextBridge:
    """One in-flight publication and one replaceable pending complete snapshot.

All methods belong to the same event loop. offer() never awaits validity or I/O.
The validity callback runs only in the background; it is not an atomic write
guard. Integration must retain committed-root/source guards at actual provider
writes. Updates, including guard_binding, are passed through without rebuilding.

A fence is permanent for this instance. Late results cannot reopen it or advance
receipt state. Receipt status describes transport evidence, never comprehension.
"""

    def __init__(
        self, route: VoiceRoute, initial_context: VoiceContextUpdate, *,
        validity: ContextValidity, publish_timeout: float = 2.0,
        clock: Clock = utc_now, on_failure: Callable[[str], None] | None = None,
        preflight_expiry_refresh: bool = False,
    ) -> None:
        if not math.isfinite(publish_timeout) or publish_timeout <= 0:
            raise ValueError("Publication timeout must be finite and positive.")
        if initial_context.route != route.identity or initial_context.mode != "snapshot":
            raise ValueError("Initial context must be a snapshot bound to this route.")
        if initial_context.expires_at is not None and initial_context.expires_at <= clock():
            raise ValueError("Initial context has expired.")
        self.identity = initial_context.route
        self._route = route
        self._validity = validity
        self._timeout = publish_timeout
        self._clock = clock
        self._on_failure = on_failure
        # Opt-in requires a caller that gates speech independently from context
        # publication. Other routes retain the existing fail-closed behavior.
        self._expiry_refresh_enabled = preflight_expiry_refresh
        self._awaiting_fresh_snapshot = False
        self._fresh_context = asyncio.Event()
        self._fresh_context.set()
        self._latest = initial_context
        self._published_context = initial_context
        self._retry_unwritten = False
        self._deferred_since_write = False
        self._pending: VoiceContextUpdate | None = None
        self._last_receipt: ContextReceipt | None = None
        self._wake = asyncio.Event()
        self._worker: asyncio.Task | None = None
        self._active_publication = None
        self._publication_settled = asyncio.Event()
        self._publication_settled.set()
        self._quiescing = 0
        self._normal_turn_refill = False
        self._jobs: set[asyncio.Task] = set()
        self._generation = 0
        self._fenced = False
        self._closed = False
        self._reason: str | None = None
        self._diagnostic_count = 0
        self._diagnostic_truncated = False

    @property
    def publication_in_flight(self) -> bool:
        return not self._closed and not self._fenced and not self._publication_settled.is_set()

    @property
    def awaiting_fresh_context(self) -> bool:
        return (not self._fenced and not self._closed and
                (self._awaiting_fresh_snapshot or
                 getattr(self._route, "context_reconciliation_pending", False) is True or
                 self._expiry_refresh_enabled and self._expired(self._published_context)))

    async def wait_for_fresh_context(self, timeout: float) -> bool:
        """Caller-owned wait, not a new publication task or a provider retry."""
        if self.awaiting_fresh_context:
            self._fresh_context.clear()
        try:
            async with asyncio.timeout(timeout):
                await self._fresh_context.wait()
        except TimeoutError:
            return False
        return not self._closed and not self._fenced and not self.awaiting_fresh_context

    @property
    def status(self) -> ContextBridgeStatus:
        return ContextBridgeStatus(
            self.identity, self._worker is not None and not self._worker.done(),
            self._fenced, self._reason, self._latest.cursor, self._last_receipt,
            self._latest.update_id if self._retry_unwritten else None,
            self._latest.update_id if (not self._fenced and not self._closed
                and self._active_publication is None and self._pending is None
                and not self._jobs and self._can_refill(self._latest)) else None,
        )

    def note_normal_turn_clear(self):
        # An explicit foreground clear is distinct from a rejected publication.
        # This enables one guarded refill offer, not retries of failed candidates.
        if (not self._closed and not self._fenced
                and getattr(self._route, "response_context_empty_proof", None) is not None):
            self._normal_turn_refill = True

    def _can_refill(self, update):
        refill = getattr(self._route, "can_refill_context", None)
        return (self._normal_turn_refill and not self._expired(update)
                and callable(refill) and refill(update) is True)

    async def quiesce_for_normal_turn(self) -> bool:
        """Join optional work before a request takes ownership of provider context.

        Cancel only preflight, which cannot have mutated the provider. A writer
        already entered retains its original deadline and must settle normally.
        Caller timeout never implies rollback or authorizes an unknown outcome.
        """
        if self._closed or self._fenced:
            return False
        self._quiescing += 1
        try:
            if self._pending is not None:
                self._pending = None
                self._retry_unwritten = True
            active = self._active_publication
            if active is not None:
                job, progress = active
                if not job.done() and not progress.adapter_entered:
                    progress.normal_turn_pause = True
                    progress.write_evidence.close()
                    job.cancel()
            await self._publication_settled.wait()
            return not self._closed and not self._fenced
        finally:
            self._quiescing -= 1

    def _park_normal_preflight(self, update, progress):
        if not self._expiry_refresh_enabled or not progress.normal_turn_pause or progress.adapter_entered:
            return False
        self._awaiting_fresh_snapshot = True
        self._fresh_context.clear()
        self._retry_unwritten = self._retry_unwritten or update == self._latest
        self._deferred_since_write = False
        self._reason = "context_normal_turn_awaiting_snapshot"
        return True

    async def start(self) -> None:
        """Initial context was supplied to route.start(); do not publish it twice."""
        if self._closed or self._fenced:
            raise RuntimeError("A fenced bridge cannot be restarted.")
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    def offer(self, update: VoiceContextUpdate) -> ContextOfferResult:
        if self._closed or self._fenced or self._worker is None or self._worker.done():
            return ContextOfferResult(OfferDisposition.REJECTED, "publication_not_open")
        if self._quiescing:
            return ContextOfferResult(OfferDisposition.REJECTED, "publication_quiescing")
        if update.route != self.identity:
            return ContextOfferResult(OfferDisposition.REJECTED, "route_mismatch")
        if update.mode != "snapshot":
            return ContextOfferResult(OfferDisposition.REJECTED, "delta_not_supported")
        if self._expired(update):
            return ContextOfferResult(OfferDisposition.REJECTED, "context_expired")
        if update.cursor.epoch != self._latest.cursor.epoch:
            return ContextOfferResult(OfferDisposition.REJECTED, "epoch_requires_new_route")
        if update.cursor.revision < self._latest.cursor.revision:
            return ContextOfferResult(OfferDisposition.REJECTED, "older_cursor")
        if update.cursor.revision == self._latest.cursor.revision:
            if update == self._latest:
                if (self._active_publication is None and self._pending is None
                        and self._can_refill(update)):
                    self._retry_unwritten = self._deferred_since_write = False
                    self._normal_turn_refill = False
                    self._pending = update
                    self._reason = None
                    self._wake.set()
                    return ContextOfferResult(OfferDisposition.QUEUED, "refill_acknowledged_empty")
                if self._retry_unwritten:
                    self._retry_unwritten = False
                    self._pending = update
                    self._reason = None
                    self._wake.set()
                    return ContextOfferResult(OfferDisposition.QUEUED, "retry_confirmed_unwritten")
                return ContextOfferResult(OfferDisposition.DUPLICATE)
            return ContextOfferResult(OfferDisposition.REJECTED, "conflicting_cursor")
        disposition = OfferDisposition.COALESCED if self._pending is not None else OfferDisposition.QUEUED
        self._latest = update
        self._normal_turn_refill = False
        self._retry_unwritten = False
        self._reason = None
        self._pending = update
        self._wake.set()
        return ContextOfferResult(disposition)

    def fence(self, reason: str) -> None:
        """Synchronous local fence, not proof that a provider write was rolled back."""
        if not self._fenced:
            self._fenced = True
            self._generation += 1
            self._reason = reason
        self._pending = None
        self._retry_unwritten = False
        self._normal_turn_refill = False
        self._last_receipt = None
        self._fresh_context.set()
        for job in tuple(self._jobs):
            job.cancel()
        self._wake.set()

    async def close(self) -> None:
        """Join owned work; an uncooperative adapter cannot be declared stopped."""
        self._closed = True
        self.fence("bridge_closed")
        tasks = tuple(self._jobs)
        if self._worker is not None:
            self._worker.cancel()
            tasks += (self._worker,)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._worker = None

    def _expired(self, update: VoiceContextUpdate) -> bool:
        return update.expires_at is not None and update.expires_at <= self._clock()

    def _current(self, token: int) -> bool:
        return not self._closed and not self._fenced and token == self._generation

    def _finished(self, task: asyncio.Task) -> None:
        self._jobs.discard(task)
        if not task.cancelled():
            task.exception()

    def _fail(self, reason: str) -> None:
        if self._closed or self._fenced:
            return
        self.fence(reason)
        if self._on_failure is not None:
            self._on_failure(reason)

    def _diagnostic(self, update, progress, stage, outcome):
        """Bounded stage evidence, not a provider write or comprehension receipt."""
        if stage not in _PUBLICATION_STAGES or outcome not in _PUBLICATION_OUTCOMES:
            return
        try:
            if self._diagnostic_count >= _PUBLICATION_LOG_LIMIT:
                if self._diagnostic_truncated:
                    return
                self._diagnostic_truncated = True
                record = {"stage": "truncated", "route_id": str(self.identity.route_id),
                          "route_generation": self.identity.route_generation,
                          "record_limit": _PUBLICATION_LOG_LIMIT,
                          "absence_after_limit_is_inconclusive": True}
            else:
                self._diagnostic_count += 1
                now = asyncio.get_running_loop().time()
                record = {
                    "route_id": str(self.identity.route_id),
                    "route_generation": self.identity.route_generation,
                    "update_id": str(update.update_id),
                    "cursor_revision": update.cursor.revision,
                    "stage": stage, "outcome": outcome,
                    "phase": (progress.phase if progress.phase in {"preflight", "validity", "adapter"}
                              else "unrecognized_code"),
                    "elapsed_ms": round(1000 * (now - progress.started_at), 3),
                    "phase_elapsed_ms": round(1000 * (now - progress.phase_started_at), 3),
                    "deadline_ms": 1000 * self._timeout,
                    "budget_phase": "preparation" if progress.mutation_started_at is None else "mutation",
                    "preparation_budget_ms": 1000 * self._timeout,
                    "mutation_budget_ms": 1000 * self._timeout,
                    "mutation_started_monotonic_s": progress.mutation_started_at,
                    "budget_remaining_ms": round(1000 * (
                        (progress.started_at if progress.mutation_started_at is None
                         else progress.mutation_started_at) + self._timeout - now), 3),
                    "adapter_entered": progress.adapter_entered,
                    "write_state": progress.write_evidence.state,
                    "owned_work_joined": progress.write_evidence.work_joined,
                    "first_possible_operation": progress.write_evidence.first_operation,
                    "deleted_context_acknowledged": progress.write_evidence.deleted_context_acknowledged,
                    "replacement_create_possible": progress.write_evidence.replacement_create_possible,
                    "created_context_acknowledged": progress.write_evidence.created_context_acknowledged,
                    "superseded": update.cursor.revision < self._latest.cursor.revision,
                }
            _PUBLICATION_LOGGER.warning("voice_context_publication_diagnostic %s",
                                        json.dumps(record, sort_keys=True, separators=(",", ":")))
        except Exception:
            # Diagnostics must never change admission, cancellation or fencing.
            pass

    def _write_budget(self, update, token, progress):
        """Veto late factories; start the mutation/ACK budget exactly once.

        Guard acquisition and writer queueing remain inside the original finite
        preparation budget. Only an instrumented factory can start the second
        budget. Delete/create never receive independent resetting allowances.
        This is not source admission, send completion or a provider ACK.
        """
        if not self._current(token):
            return False
        now = asyncio.get_running_loop().time()
        if progress.mutation_started_at is None:
            if now >= progress.started_at + self._timeout:
                return False
            progress.mutation_started_at = now
            self._diagnostic(update, progress, "mutation_budget_entry", "started")
        return now < progress.mutation_started_at + self._timeout

    async def _wait_publication(self, job, progress):
        loop = asyncio.get_running_loop()
        done, _ = await asyncio.wait({job}, timeout=max(
            0.0, progress.started_at + self._timeout - loop.time()))
        if not done and progress.mutation_started_at is not None:
            # No new work, replay, or per-operation renewal. The first writer
            # factory must have crossed its boundary before preparation expired.
            done, _ = await asyncio.wait({job}, timeout=max(
                0.0, progress.mutation_started_at + self._timeout - loop.time()))
        return done

    async def _publish(self, update: VoiceContextUpdate, token: int, progress: _PublicationProgress):
        if not self._current(token) or update.cursor.revision < self._latest.cursor.revision:
            return None
        if self._expired(update):
            raise _PublicationRejected("context_expired_before_write")
        progress.phase = "validity"
        progress.phase_started_at = asyncio.get_running_loop().time()
        self._diagnostic(update, progress, "validity_entry", "started")
        try:
            if self._deferred_since_write and not self._can_refill(update):
                previous = self._published_context
                retain = getattr(self._route, "can_retain_context", None)
                if (not callable(retain) or self._expired(previous) or retain(previous) is not True
                        or await self._validity(previous) is not True
                        or self._expired(previous) or retain(previous) is not True):
                    raise _PublicationRejected("retained_context_unavailable_before_retry")
            valid = await self._validity(update)
        except asyncio.CancelledError:
            self._diagnostic(update, progress, "validity_exit", "cancelled")
            raise
        except Exception:
            self._diagnostic(update, progress, "validity_exit", "failed")
            raise
        self._diagnostic(update, progress, "validity_exit", "valid" if valid is True else "invalid")
        # Validation may await while a newer complete snapshot replaces this
        # one. An obsolete result must not fence that newer publication.
        if not self._current(token) or update.cursor.revision < self._latest.cursor.revision:
            return None
        if self._expired(update):
            raise _PublicationRejected("context_expired_before_write")
        if valid is not True:
            raise _PublicationRejected("context_validity_rejected")
        if progress.normal_turn_pause:
            raise asyncio.CancelledError
        progress.phase = "adapter"
        progress.phase_started_at = asyncio.get_running_loop().time()
        # Set synchronously before calling the adapter. Once entered, even a
        # superseded timeout cannot be treated as proof of no provider effect.
        progress.adapter_entered = True
        self._diagnostic(update, progress, "adapter_entry", "started")
        try:
            tracked = getattr(self._route, "publish_context_tracked", None)
            if callable(tracked):
                receipt = await tracked(update, write_evidence=progress.write_evidence)
            else:
                receipt = await self._route.publish_context(update)
        except asyncio.CancelledError:
            self._diagnostic(update, progress, "adapter_return", "cancelled")
            raise
        except Exception:
            self._diagnostic(update, progress, "adapter_return", "failed")
            raise
        self._diagnostic(update, progress, "adapter_return", "returned")
        return receipt

    def _hold_expired_snapshot(self, update, progress, *, preflight=False,
                               after_acknowledged_delete=False, after_acknowledged_create=False) -> bool:
        """Drop an expired candidate only with a resolved publication outcome.

        Adapter entry alone does not imply a mutation, but recovery after entry
        requires sealed, joined evidence: either no write or a matching deleted
        ACK with no possible create. The latter is NOT an unwritten attempt.
        Neither path retries the old candidate or any uncertain mutation.
        """
        if after_acknowledged_create:
            resolved = progress.adapter_entered and progress.write_evidence.confirmed_created
            outcome = "expired_after_acknowledged_create"
        elif after_acknowledged_delete:
            resolved = progress.adapter_entered and progress.write_evidence.confirmed_deleted_without_create
            outcome = "expired_after_acknowledged_delete"
        else:
            resolved = (not progress.adapter_entered if preflight else
                        progress.adapter_entered and progress.write_evidence.proven_unwritten)
            outcome = "expired_preflight" if preflight else "expired_unwritten_adapter"
        if not self._expiry_refresh_enabled or not self._expired(update) or not resolved:
            return False
        self._awaiting_fresh_snapshot = True
        self._fresh_context.clear()
        self._retry_unwritten = False
        self._deferred_since_write = False
        self._reason = "context_expired_awaiting_snapshot"
        self._diagnostic(update, progress, "context_refresh_wait", outcome)
        return True

    async def _defer_unwritten(self, update, token, progress) -> bool:
        """Retain only exact, acknowledged, still-valid prior context.

        This runs after publication has terminated and its children have joined.
        No provider work receives extra deadline time. A retry needs an explicit
        offer (or an already pending newer snapshot), never a spinning backlog.
        """
        if self.awaiting_fresh_context:
            # Never resume speech by retaining an old snapshot during expiry
            # recovery. Only a fresh, guarded replacement can end this hold.
            return False
        tracked = callable(getattr(self._route, "publish_context_tracked", None))
        no_write = progress.write_evidence.proven_unwritten or (
            tracked and not progress.adapter_entered)
        retain = getattr(self._route, "can_retain_context", None)
        previous = self._published_context
        if not self._current(token) or not no_write or not callable(retain):
            return False
        try:
            if self._expired(previous) or retain(previous) is not True:
                return False
        except Exception:
            return False
        self._diagnostic(update, progress, "retained_context_validity_entry", "started")
        check = asyncio.create_task(self._validity(previous))
        self._jobs.add(check)
        check.add_done_callback(self._finished)
        done, _ = await asyncio.wait({check}, timeout=self._timeout)
        if not done:
            check.cancel()
            await asyncio.wait({check}, timeout=min(self._timeout, _CANCELLATION_JOIN_LIMIT_SECONDS))
            self._diagnostic(update, progress, "retained_context_validity_exit", "failed")
            return False
        try:
            valid = (check.result() is True and self._current(token)
                     and not self._expired(previous) and retain(previous) is True)
        except (Exception, asyncio.CancelledError):
            valid = False
        self._diagnostic(update, progress, "retained_context_validity_exit", "valid" if valid else "invalid")
        if not valid:
            return False
        self._retry_unwritten = update == self._latest
        self._deferred_since_write = True
        self._reason = "context_publication_deferred_unwritten"
        self._diagnostic(update, progress, "publication_deferred", "proven_unwritten")
        return True

    def _hold_task_progress(self, update, progress, *, after_acknowledged_delete=False,
                            control_change=False) -> bool:
        """Verified task progress needs new metadata, never stale context reuse.

        Preflight is unwritten. After adapter entry, only a sealed/joined exact
        delete ACK with no possible create and current empty-slot ownership is
        resolved. Neither case replays a mutation or permits old metadata.
        The existing publisher notices/tick and task projector supply a newer
        snapshot; existing response-owner waits remain finite and cancellable.
        """
        if not self._expiry_refresh_enabled:
            return False
        if after_acknowledged_delete:
            current = getattr(self._route, "can_await_fresh_context_after_delete", None)
            if (not progress.adapter_entered or not progress.write_evidence.confirmed_deleted_without_create
                    or not callable(current)):
                return False
            try:
                if current(update, write_evidence=progress.write_evidence) is not True:
                    return False
            except Exception:
                return False
        elif progress.adapter_entered or progress.phase != "validity":
            return False
        self._awaiting_fresh_snapshot = True
        self._fresh_context.clear()
        self._retry_unwritten = self._deferred_since_write = False
        self._reason = ("context_task_control_awaiting_snapshot" if control_change
                        else "context_task_progress_awaiting_snapshot")
        self._diagnostic(update, progress, "context_refresh_wait",
                         ("task_control_after_acknowledged_delete" if after_acknowledged_delete
                          else "task_control_unwritten") if control_change else
                         ("task_progress_after_acknowledged_delete" if after_acknowledged_delete
                          else "task_progress_unwritten"))
        return True

    def _hold_owner_busy(self, update, progress) -> bool:
        """Suspend speech without treating a busy owner as revoked authority.

        Only unwritten work can use the existing bounded explicit-offer retry.
        An acknowledged delete instead requires a fresh snapshot and exact
        empty-slot ownership. Neither path authorizes old context or reopens a
        fence; every later publication runs all canonical guards again.
        """
        if not self._expiry_refresh_enabled:
            return False
        unwritten = (progress.write_evidence.proven_unwritten if progress.adapter_entered
                     else progress.phase == "validity")
        if not unwritten:
            current = getattr(self._route, "can_await_fresh_context_after_delete", None)
            if (not progress.adapter_entered or not progress.write_evidence.confirmed_deleted_without_create
                    or not callable(current)):
                return False
            try:
                if current(update, write_evidence=progress.write_evidence) is not True:
                    return False
            except Exception:
                return False
        self._awaiting_fresh_snapshot = True
        self._fresh_context.clear()
        self._retry_unwritten = unwritten and update == self._latest
        self._deferred_since_write = False
        self._reason = "context_owner_busy_awaiting_snapshot"
        self._diagnostic(update, progress, "context_refresh_wait",
            "owner_busy_unwritten" if unwritten else "owner_busy_after_acknowledged_delete")
        return True

    def _hold_unavailable_after_delete(self, update, progress) -> bool:
        """Wait without speech for new authorized context, never retry a mutation.

        Unlike expiry, runtime unavailability does not prove why local state is
        unusable. It may require reconstruction. Only a future independently
        guarded snapshot can release this hold; current-source/retirement checks
        remain authoritative. Unknown writes, unjoined work and unconfirmed
        deletes cannot enter it, even if a receipt claims local unavailability.
        """
        if not self._expiry_refresh_enabled or not progress.adapter_entered:
            return False
        inherited = False
        if not progress.write_evidence.confirmed_deleted_without_create:
            current = getattr(self._route, "can_await_fresh_context_after_delete", None)
            if (not self._awaiting_fresh_snapshot or not progress.write_evidence.proven_unwritten
                    or not callable(current)):
                return False
            try:
                inherited = current(update, write_evidence=progress.write_evidence) is True
            except Exception:
                return False
            if not inherited:
                return False
        self._awaiting_fresh_snapshot = True
        self._fresh_context.clear()
        self._retry_unwritten = False
        self._deferred_since_write = False
        self._reason = "context_source_unavailable_awaiting_snapshot"
        self._diagnostic(update, progress, "context_refresh_wait",
                         "unavailable_in_acknowledged_empty_context" if inherited
                         else "unavailable_after_acknowledged_delete")
        return True

    def _hold_empty_after_deadline(self, update, progress) -> bool:
        """An owned ACKed delete is resolved, even when authority checking times out.

        The slot is empty, not usable. Only a new snapshot's unchanged source
        guard can authorize new content; neither this candidate nor its delete
        is retried. Adapters without exact local ownership proof fail closed.
        """
        current = getattr(self._route, "can_await_fresh_context_after_delete", None)
        if (not self._expiry_refresh_enabled or not progress.adapter_entered
                or not progress.write_evidence.confirmed_deleted_without_create
                or not callable(current)):
            return False
        try:
            if current(update, write_evidence=progress.write_evidence) is not True:
                return False
        except Exception:
            return False
        self._awaiting_fresh_snapshot = True
        self._fresh_context.clear()
        self._retry_unwritten = False
        self._deferred_since_write = False
        self._reason = "context_empty_awaiting_snapshot"
        self._diagnostic(update, progress, "context_refresh_wait", "deadline_after_acknowledged_delete")
        return True

    async def _run(self) -> None:
        while not self._closed and not self._fenced:
            # Freshness is availability, not withdrawal. Individual callers and
            # publications keep bounded waits; an idle healthy route does not
            # self-retire because a fresh observation has not arrived yet.
            await self._wake.wait()
            self._wake.clear()
            if self._closed or self._fenced:
                return
            update, self._pending = self._pending, None
            if update is None:
                continue
            self._publication_settled.clear()
            try:
                await self._publish_one(update)
            finally:
                self._active_publication = None
                self._publication_settled.set()

    async def _publish_one(self, update):
        token = self._generation
        started = asyncio.get_running_loop().time()
        progress = _PublicationProgress(started_at=started, phase_started_at=started)
        progress.write_evidence = ContextWriteEvidence(
            write_budget=lambda update=update, token=token, progress=progress:
                self._write_budget(update, token, progress))
        job = asyncio.create_task(self._publish(update, token, progress))
        self._jobs.add(job)
        job.add_done_callback(self._finished)
        self._active_publication = (job, progress)
        done = await self._wait_publication(job, progress)
        if not done:
            obsolete_preflight = (self._current(token) and not progress.adapter_entered
                                  and update.cursor.revision < self._latest.cursor.revision)
            tracked = callable(getattr(self._route, "publish_context_tracked", None))
            outcome = ("superseded_preflight" if obsolete_preflight else
                       "cancellation_pending" if tracked and progress.write_evidence.state != "write_possible"
                       else "failed_closed")
            self._diagnostic(update, progress, "publication_deadline", outcome)
            progress.write_evidence.close()
            job.cancel()
            if obsolete_preflight or callable(getattr(self._route, "publish_context_tracked", None)):
                # No overlapping replacement: first join cancellation of
                # the owned preflight. This is not extra publication time.
                self._diagnostic(update, progress, "cancellation_join_entry", "started")
                joined, _ = await asyncio.wait({job},
                    timeout=min(self._timeout, _CANCELLATION_JOIN_LIMIT_SECONDS))
                if joined:
                    self._diagnostic(update, progress, "cancellation_join_return", "terminal")
                    if not self._current(token):
                        return
                    if obsolete_preflight and not progress.adapter_entered:
                        self._diagnostic(update, progress, "publication_skipped", "superseded_preflight")
                        return
                    # A hard source/scope failure racing the timeout must not
                    # be mistaken for cancellation. Require the actual job's
                    # cancelled terminal state as well as sealed/joined proof.
                    if job.cancelled() and self._hold_empty_after_deadline(update, progress):
                        return
                    if await self._defer_unwritten(update, token, progress):
                        return
                else:
                    self._diagnostic(update, progress, "cancellation_join_deadline", "cleanup_unconfirmed")
                # Uncooperative work stays owned by _jobs/close(). Never
                # launch more jobs or claim that cancellation stopped it.
            self._diagnostic(update, progress, "publication_fenced", "failed_closed")
            self._fail("context_publication_deadline")
            return
        if not self._current(token):
            return
        try:
            receipt = job.result()
        except _PublicationRejected as error:
            if (str(error) == "context_expired_before_write"
                    and self._hold_expired_snapshot(update, progress, preflight=True)):
                return
            if await self._defer_unwritten(update, token, progress):
                return
            self._fail(str(error))
            return
        except asyncio.CancelledError:
            if job.cancelled() and self._park_normal_preflight(update, progress):
                return
            if await self._defer_unwritten(update, token, progress):
                return
            self._fail("context_publication_cancelled")
            return
        except Exception as error:
            if isinstance(error, CanonicalOwnerBusy) and self._hold_owner_busy(update, progress):
                return
            if (isinstance(error, (TaskSnapshotProgress, TaskSnapshotControlChange))
                    and self._hold_task_progress(update, progress,
                        control_change=isinstance(error, TaskSnapshotControlChange))):
                return
            # The canonical guard can expire after validity, before the
            # adapter's first mutation. Only its typed expiry rejection is
            # recoverable; withdrawal, deadlines and arbitrary failures are
            # not reclassified merely because the candidate is also old.
            if (isinstance(error, SourceError) and error.code == "voice_context_expired"
                    and self._hold_expired_snapshot(update, progress)):
                return
            if await self._defer_unwritten(update, token, progress):
                return
            self._fail("context_publication_failed")
            return
        if receipt is None:
            return
        if (not isinstance(receipt, ContextReceipt) or receipt.route != self.identity
                or receipt.update_id != update.update_id or receipt.cursor != update.cursor):
            self._fail("context_receipt_mismatch_or_unsupported")
            return
        if receipt.status in {"rejected", "failed", "unknown"}:
            if (receipt.status == "rejected" and receipt.reason == "context_owner_busy"
                    and self._hold_owner_busy(update, progress)):
                return
            if (receipt.status == "rejected" and receipt.reason == "context_task_control_changed"
                    and self._hold_task_progress(update, progress, after_acknowledged_delete=True,
                                                control_change=True)):
                return
            if (receipt.status == "rejected" and receipt.reason == "context_task_progress"
                    and self._hold_task_progress(update, progress, after_acknowledged_delete=True)):
                return
            if (receipt.status == "rejected" and receipt.reason == "context_acknowledged_stale"
                    and self._hold_expired_snapshot(update, progress, after_acknowledged_create=True)):
                return
            if (receipt.status == "rejected" and receipt.reason == "context_source_unavailable"
                    and self._hold_unavailable_after_delete(update, progress)):
                return
            if (receipt.status == "rejected" and receipt.reason == "context_not_current_snapshot"
                    and (self._hold_expired_snapshot(update, progress)
                         or self._hold_expired_snapshot(update, progress, after_acknowledged_delete=True))):
                return
            if (receipt.status == "rejected" and receipt.reason == "context_response_active"
                    and progress.write_evidence.proven_unwritten):
                self._retry_unwritten = update == self._latest
                self._reason = "context_publication_deferred_response"
                return
            if await self._defer_unwritten(update, token, progress):
                return
            self._fail("context_receipt_unconfirmed")
            return
        self._last_receipt = receipt
        self._reason = None
        if receipt.status in {"written", "provider_acknowledged"}:
            self._published_context = update
            self._deferred_since_write = False
            if self._expired(update) and self._expiry_refresh_enabled:
                self._awaiting_fresh_snapshot = True
                self._fresh_context.clear()
                self._reason = "context_expired_awaiting_snapshot"
            else:
                refreshed = self._awaiting_fresh_snapshot
                self._awaiting_fresh_snapshot = False
                self._fresh_context.set()
                if refreshed:
                    self._diagnostic(update, progress, "context_refresh_return", "fresh_snapshot_written")
