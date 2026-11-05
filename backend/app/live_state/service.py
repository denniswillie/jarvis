"""One event-loop-owned Live State writer; durable authorities live elsewhere."""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable
from uuid import UUID, uuid4

from app.live_state.models import EntryUpserted, LiveEvent, LiveSnapshot, SessionScope, StateCursor
from app.live_state.reducer import Reduction, expire, fence, initial_state, reduce_event, register_producer


@dataclass(frozen=True, slots=True)
class Submission:
    queued: bool
    reason: str


@dataclass(frozen=True, slots=True)
class StateNotice:
    snapshot: LiveSnapshot
    reason: str
    resync_required: bool
    critical: bool = False


class StateSubscription:
    """Overflow is explicit; consumers must retire/resync, not skip invalidation."""

    def __init__(self, service: LiveStateService, capacity: int):
        self._service = service
        self._queue: asyncio.Queue[StateNotice | None] = asyncio.Queue(maxsize=capacity)
        self._needs_resync = False
        self._closed = False

    def _offer(self, notice: StateNotice) -> None:
        if self._closed:
            return
        if self._queue.full() or self._needs_resync:
            self._needs_resync = True
            while not self._queue.empty():
                self._queue.get_nowait()
            notice = replace(notice, reason="subscriber_resynchronization_required",
                             resync_required=True, critical=True)
        self._queue.put_nowait(notice)

    def acknowledge_resync(self, cursor: StateCursor) -> bool:
        self._service._require_owner_loop()
        if self._closed or self._service.snapshot().cursor != cursor or self._service.needs_resync:
            return False
        self._needs_resync = False
        return True

    async def receive(self) -> StateNotice | None:
        if self._closed and self._queue.empty():
            return None
        return await self._queue.get()

    def close(self) -> None:
        self._service._require_owner_loop()
        if self._closed:
            return
        self._closed = True
        self._service._subscribers.discard(self)
        while not self._queue.empty():
            self._queue.get_nowait()
        self._queue.put_nowait(None)


class LiveStateService:
    """Compose one instance for the owner's active scope in the initial API process.

    `submit` only acknowledges bounded enqueue, never durable acceptance. Producer
    registration, readiness and reset are trusted composition APIs, not LLM tools.
    External workers wake canonical adapters; they do not mutate this instance.
    """

    def __init__(self, scope: SessionScope, *, clock: Callable[[], datetime], queue_capacity: int = 256):
        if not 1 <= queue_capacity <= 256:
            raise ValueError("Queue capacity must be between 1 and 256.")
        self._clock = clock
        self._state = initial_state(scope, epoch=uuid4(), now=clock())
        self._capacity = queue_capacity
        self._inbox: deque[LiveEvent] = deque()
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker: asyncio.Task | None = None
        self._closed = False
        self._subscribers: set[StateSubscription] = set()
        self.last_rejection: str | None = None

    @property
    def needs_resync(self) -> bool:
        return self._state.needs_resync

    def snapshot(self) -> LiveSnapshot:
        # No I/O, mutable dictionary or model call escapes through this accessor.
        return self._state.snapshot

    def _require_owner_loop(self) -> None:
        if self._loop is None or asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("Live State mutation requires its owning event loop.")

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("Closed Live State cannot restart; reconstruct it from authorities.")
        if self._worker is not None:
            self._require_owner_loop()
            return
        self._loop = asyncio.get_running_loop()
        self._worker = asyncio.create_task(self._run(), name="jarvis-live-state")

    def _commit(self, result: Reduction) -> None:
        if not result.applied:
            self.last_rejection = result.reason
            return
        self._state = result.state
        notice = StateNotice(self.snapshot(), result.reason, self.needs_resync, result.critical)
        for subscriber in tuple(self._subscribers):
            subscriber._offer(notice)

    def register_producer(self, producer_id: str, epoch: UUID) -> None:
        self._require_owner_loop()
        if self._closed:
            raise RuntimeError("Live State is closed.")
        self._commit(register_producer(self._state, producer_id, epoch, now=self._clock()))

    def submit(self, event: LiveEvent) -> Submission:
        self._require_owner_loop()
        if self._closed:
            return Submission(False, "closed")
        if self.needs_resync:
            return Submission(False, "resynchronization_required")
        if event.scope != self.snapshot().scope:
            return Submission(False, "scope_mismatch")
        if dict(self._state.producer_epochs).get(event.producer.producer_id) != event.producer.producer_epoch:
            return Submission(False, "unregistered_or_retired_producer")
        if len(self._inbox) >= self._capacity:
            # Never erase an intervening withdrawal/correction to make room.
            if isinstance(event, EntryUpserted) and self._inbox:
                previous = self._inbox[-1]
                if (isinstance(previous, EntryUpserted)
                        and previous.producer.producer_id == event.producer.producer_id
                        and previous.producer.producer_epoch == event.producer.producer_epoch
                        and previous.producer.aggregate.namespace == event.producer.aggregate.namespace
                        and previous.producer.aggregate.aggregate_id == event.producer.aggregate.aggregate_id
                        and previous.entry.entry_id == event.entry.entry_id
                        and previous.entry.sources == event.entry.sources
                        and previous.entry.validity == event.entry.validity == "active"):
                    if event.producer.aggregate.revision <= previous.producer.aggregate.revision:
                        return Submission(False, "outdated_queued_update")
                    self._inbox[-1] = event
                    self._wake.set()
                    return Submission(True, "coalesced")
            self._inbox.clear()
            self._idle.set()
            self._commit(fence(self._state, now=self._clock(), reason="inbox_overflow"))
            return Submission(False, "resynchronization_required")
        self._inbox.append(event)
        self._idle.clear()
        self._wake.set()
        return Submission(True, "queued")

    def subscribe(self, *, after: StateCursor | None = None, capacity: int = 8) -> StateSubscription:
        self._require_owner_loop()
        if self._closed or not 1 <= capacity <= 32 or len(self._subscribers) >= 64:
            raise RuntimeError("Subscription unavailable or capacity exceeded.")
        subscription = StateSubscription(self, capacity)
        self._subscribers.add(subscription)
        mismatch = after is not None and after != self.snapshot().cursor
        subscription._needs_resync = mismatch
        subscription._offer(StateNotice(self.snapshot(), "initial_snapshot", mismatch or self.needs_resync, mismatch))
        return subscription

    def mark_ready(self) -> None:
        self._require_owner_loop()
        if self._closed or self.needs_resync or self._inbox:
            raise RuntimeError("Finish canonical reconstruction before marking ready.")
        snapshot = self.snapshot()
        ready = LiveSnapshot(**(snapshot.model_dump() | {
            "readiness": "ready", "as_of": self._clock(),
            "cursor": StateCursor(epoch=snapshot.cursor.epoch, revision=snapshot.cursor.revision + 1),
        }))
        self._commit(Reduction(replace(self._state, snapshot=ready), True, "canonical_baseline_ready"))

    def reset_for_reconciliation(self) -> None:
        """Caller must retire consumer context, register NEW epochs and reread owners."""
        self._require_owner_loop()
        if self._closed:
            raise RuntimeError("Live State is closed.")
        self._inbox.clear()
        self._idle.set()
        self._state = initial_state(self.snapshot().scope, epoch=uuid4(), now=self._clock())
        for subscriber in tuple(self._subscribers):
            subscriber._needs_resync = True
            subscriber._offer(StateNotice(self.snapshot(), "scope_reconstructed", True, True))

    async def wait_idle(self) -> None:
        self._require_owner_loop()
        while True:
            await self._idle.wait()
            # Event.wait() remembers a set even if another producer clears it
            # before this waiter resumes. Recheck the actual queue on our owner
            # loop so callers cannot mistake an old drain for current readiness.
            # There is no await between this check and returning to the caller.
            if not self._inbox:
                return

    async def _run(self) -> None:
        try:
            while not self._closed:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                except TimeoutError:
                    pass
                self._wake.clear()
                self._commit(expire(self._state, now=self._clock()))
                while self._inbox and not self._closed:
                    event = self._inbox.popleft()
                    self._commit(reduce_event(self._state, event, now=self._clock()))
                    if self.needs_resync:
                        self._inbox.clear()
                    # One selected memory can emit several critical source
                    # confirmations before its upsert. Let ready subscribers
                    # consume each notice instead of overflowing their bounded
                    # queues merely because this writer owns the event loop.
                    # A genuinely blocked subscriber still overflows and fences;
                    # no notice, withdrawal or ordering check is coalesced away.
                    await asyncio.sleep(0)
                self._idle.set()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._inbox.clear()
            self._commit(fence(self._state, now=self._clock(), reason="reducer_failed"))
            raise
        finally:
            self._idle.set()

    async def close(self) -> None:
        self._require_owner_loop()
        if self._closed:
            return
        self._closed = True
        self._inbox.clear()
        self._idle.set()
        self._commit(fence(self._state, now=self._clock(), reason="service_closed"))
        for subscriber in tuple(self._subscribers):
            subscriber.close()
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
