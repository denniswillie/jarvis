"""Pure, bounded projection reduction. No I/O or model-selected authority."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from uuid import UUID

from app.live_state.models import (
    CapabilitiesChanged, EntryRetracted, EntryUpserted, LiveEvent, LiveSnapshot,
    ProducerCursor, SessionScope, SourceDependency, SourceStateChanged,
    SourceWatermark, StateCursor, TurnRevised,
)


MAX_SEEN_EVENTS = 512
MAX_SOURCE_WATERMARKS = 512
MAX_PRODUCERS = 64


@dataclass(frozen=True, slots=True)
class ProjectionState:
    snapshot: LiveSnapshot
    producer_epochs: tuple[tuple[str, UUID], ...] = ()
    watermarks: tuple[SourceWatermark, ...] = ()
    seen_events: tuple[UUID, ...] = ()
    needs_resync: bool = False


@dataclass(frozen=True, slots=True)
class Reduction:
    state: ProjectionState
    applied: bool
    reason: str
    critical: bool = False


def _aware(now: datetime) -> None:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Projection clocks must be timezone-aware.")


def initial_state(scope: SessionScope, *, epoch: UUID, now: datetime) -> ProjectionState:
    _aware(now)
    return ProjectionState(LiveSnapshot(scope=scope, cursor=StateCursor(epoch=epoch, revision=0),
                                       as_of=now, readiness="warming"))


def _snapshot(state: ProjectionState, now: datetime, **changes) -> LiveSnapshot:
    _aware(now)
    return LiveSnapshot(**(state.snapshot.model_dump() | changes | {
        "as_of": now,
        "cursor": StateCursor(epoch=state.snapshot.cursor.epoch,
                              revision=state.snapshot.cursor.revision + 1),
    }))


def fence(state: ProjectionState, *, now: datetime, reason: str) -> Reduction:
    """Losing ordering must never silently authorize a stale projection."""
    return Reduction(replace(state, snapshot=_snapshot(state, now, entries=(), readiness="degraded"),
                             needs_resync=True), True, reason, True)


def register_producer(state: ProjectionState, producer_id: str, epoch: UUID,
                      *, now: datetime) -> Reduction:
    """Trusted composition calls this; event envelopes cannot register themselves."""
    if not producer_id or len(producer_id) > 160:
        raise ValueError("Invalid producer identity.")
    epochs = dict(state.producer_epochs)
    if epochs.get(producer_id) == epoch:
        return Reduction(state, False, "producer_already_registered")
    if producer_id not in epochs and len(epochs) >= MAX_PRODUCERS:
        return fence(state, now=now, reason="producer_capacity")
    epochs[producer_id] = epoch
    cursors = tuple(cursor for cursor in state.snapshot.producer_cursors
                    if cursor.producer_id != producer_id)
    return Reduction(replace(state, producer_epochs=tuple(sorted(epochs.items())),
                             snapshot=_snapshot(state, now, producer_cursors=cursors)),
                     True, "producer_registered")


def _source_key(source: SourceDependency):
    return source.source_namespace, source.source_id


def _source_order(source: SourceDependency) -> int:
    return source.generation if source.generation is not None else source.revision


def _cursor_key(cursor: ProducerCursor):
    return cursor.producer_id, cursor.aggregate.namespace, cursor.aggregate.aggregate_id


def dependency_available(state: ProjectionState, source: SourceDependency) -> bool:
    """Only explicit canonical adapter observations can establish availability.

    Generation-bearing authorities can validate historical content revisions at
    the current generation. Each such revision needs its own active watermark.
    """
    family = tuple(mark for mark in state.watermarks if _source_key(mark.source) == _source_key(source))
    if not family:
        return False
    if any((mark.source.generation is None) != (source.generation is None) for mark in family):
        return False
    newest = max(_source_order(mark.source) for mark in family)
    if _source_order(source) != newest:
        return False
    current = tuple(mark for mark in family if _source_order(mark.source) == newest)
    if any(mark.availability != "active" for mark in current):
        return False
    return any(mark.source == source and mark.availability == "active" for mark in current)


def expire(state: ProjectionState, *, now: datetime) -> Reduction:
    _aware(now)
    entries = tuple(entry for entry in state.snapshot.entries
                    if not any(deadline is not None and deadline <= now
                               for deadline in (entry.fresh_until, entry.expires_at)))
    if entries == state.snapshot.entries:
        return Reduction(state, False, "no_expiry")
    return Reduction(replace(state, snapshot=_snapshot(state, now, entries=entries)),
                     True, "entry_expired", True)


def reduce_event(state: ProjectionState, event: LiveEvent, *, now: datetime) -> Reduction:
    _aware(now)
    if event.scope != state.snapshot.scope:
        return Reduction(state, False, "scope_mismatch")
    if state.needs_resync:
        return Reduction(state, False, "resynchronization_required", True)
    if dict(state.producer_epochs).get(event.producer.producer_id) != event.producer.producer_epoch:
        return Reduction(state, False, "unregistered_or_retired_producer")
    if event.event_id in state.seen_events:
        return Reduction(state, False, "duplicate_event")
    cursors = {_cursor_key(cursor): cursor for cursor in state.snapshot.producer_cursors}
    key = _cursor_key(event.producer)
    previous = cursors.get(key)
    if previous is not None and event.producer.aggregate.revision <= previous.aggregate.revision:
        return Reduction(state, False, "outdated_projection_sequence")
    if previous is None and len(cursors) >= 256:
        return fence(state, now=now, reason="cursor_capacity")

    entries = {entry.entry_id: entry for entry in state.snapshot.entries
               if not any(deadline is not None and deadline <= now
                          for deadline in (entry.fresh_until, entry.expires_at))}
    changes = {}
    watermarks = state.watermarks
    critical = len(entries) != len(state.snapshot.entries)

    if isinstance(event, EntryUpserted):
        entry = event.entry
        if entry.validity != "active":
            entries.pop(entry.entry_id, None)
            critical = True
        else:
            if any(deadline is not None and deadline <= now
                   for deadline in (entry.fresh_until, entry.expires_at)):
                return Reduction(state, False, "expired_entry")
            if any(not dependency_available(state, source) for source in entry.sources):
                return Reduction(state, False, "dependency_not_confirmed")
            if entry.entry_id not in entries and len(entries) >= 128:
                return fence(state, now=now, reason="entry_capacity")
            entries[entry.entry_id] = entry
    elif isinstance(event, EntryRetracted):
        entries.pop(event.entry_id, None)
        critical = True
    elif isinstance(event, SourceStateChanged):
        mark = event.watermark
        family = tuple(old for old in watermarks if _source_key(old.source) == _source_key(mark.source))
        if any((old.source.generation is None) != (mark.source.generation is None) for old in family):
            return fence(state, now=now, reason="source_version_domain_changed")
        incoming = _source_order(mark.source)
        if family:
            latest = max(_source_order(old.source) for old in family)
            if incoming < latest:
                return Reduction(state, False, "old_source_generation")
            if incoming == latest and mark.availability == "active" and any(
                old.availability == "retracted" for old in family
            ):
                return Reduction(state, False, "withdrawal_requires_new_authority_generation")
        # Withdrawal/uncertainty applies to the family. Active historical reads
        # may coexist only at the same confirmed lifecycle generation.
        keep_family = mark.availability == "active" and all(old.availability == "active" for old in family)
        kept = tuple(old for old in watermarks if _source_key(old.source) != _source_key(mark.source)
                     or (keep_family and _source_order(old.source) == incoming
                         and old.source.revision != mark.source.revision))
        if len(kept) >= MAX_SOURCE_WATERMARKS:
            return fence(state, now=now, reason="source_watermark_capacity")
        watermarks = kept + (mark,)
        candidate = replace(state, watermarks=watermarks)
        entries = {identity: entry for identity, entry in entries.items()
                   if all(dependency_available(candidate, source) for source in entry.sources)}
        critical = True
    elif isinstance(event, TurnRevised):
        if not dependency_available(state, event.request_source):
            return Reduction(state, False, "turn_source_not_confirmed")
        changes["conversation"] = event.conversation
        critical = True
    elif isinstance(event, CapabilitiesChanged):
        changes["capabilities"] = event.capabilities
        critical = True
    else:
        raise TypeError("Unsupported Live State event.")

    cursors[key] = event.producer
    snapshot = _snapshot(state, now, entries=tuple(entries.values()),
                         producer_cursors=tuple(cursors.values()), **changes)
    return Reduction(replace(state, snapshot=snapshot, watermarks=watermarks,
                             seen_events=(state.seen_events + (event.event_id,))[-MAX_SEEN_EVENTS:]),
                     True, event.kind, critical)
