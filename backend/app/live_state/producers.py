"""Trusted adapters from canonical records to the session's disposable view.

Only backend composition supplies these records. Models, client payloads and
provider-generated identifiers cannot directly register a producer or publish
an authority watermark. Canonical versions and local projection order differ.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from app.knowledge.models import SourceVersion
from app.live_state.canonical import context_dependency, memory_dependency
from app.live_state.models import (
    AggregateRef, ConversationView, EntryRetracted, EntryUpserted,
    GuardedPayloadBinding, LiveEntry, ObservationEntry, ProducerCursor,
    SelectedMemoryEntry, SourceStateChanged, SourceWatermark, TurnRevised,
)
from app.live_state.service import LiveStateService
from app.perception.models import SessionWorkingMemoryState, VisualObservation
from app.tasks.context_events import ContextRecord


class CanonicalStateProducer:
    def __init__(self, service: LiveStateService, *, producer_id: str,
                 clock: Callable[[], datetime]):
        self.service, self.producer_id, self.clock = service, producer_id, clock
        self.epoch = uuid4()
        self._sequence = 0
        self._started = False

    def start(self):
        if self._started:
            raise RuntimeError("A producer cannot silently restart its epoch")
        self.service.register_producer(self.producer_id, self.epoch)
        self._started = True

    def _header(self):
        if not self._started:
            raise RuntimeError("Register the trusted producer before publishing")
        self._sequence += 1
        return dict(event_id=uuid4(), scope=self.service.snapshot().scope,
            emitted_at=self.clock(), producer=ProducerCursor(
                producer_id=self.producer_id, producer_epoch=self.epoch,
                aggregate=AggregateRef(namespace="live_state_projection",
                    aggregate_id=self.service.snapshot().scope.session_id,
                    revision=self._sequence)))

    def source_changed(self, watermark: SourceWatermark):
        return self.service.submit(SourceStateChanged(**self._header(), watermark=watermark))

    def publish(self, entry: LiveEntry):
        if entry.scope != self.service.snapshot().scope:
            raise ValueError("Entry belongs to another producer scope")
        if entry.sources and entry.guard_binding is None:
            raise ValueError("Canonical content needs an existing guarded payload binding")
        # Each confirmation comes from the caller's canonical read/bind, not
        # from a model asserting that it remembers an identifier.
        for dependency in entry.sources:
            receipt = self.source_changed(SourceWatermark(source=dependency,
                availability="active", recorded_at=self.clock()))
            if not receipt.queued:
                return receipt
        return self.service.submit(EntryUpserted(**self._header(), entry=entry))

    def retract(self, entry_id: str, reason: str):
        return self.service.submit(EntryRetracted(**self._header(), entry_id=entry_id, reason=reason))

    def conversation(self, record: ContextRecord):
        scope = self.service.snapshot().scope
        if (record.owner_id, record.session_id) != (scope.owner_id, scope.session_id):
            raise ValueError("Conversation record is outside the session")
        if not record.authorizes_request:
            raise ValueError("Only a finalized current request can advance the conversation")
        dependency = context_dependency(record)
        receipt = self.source_changed(SourceWatermark(source=dependency,
            availability="active", recorded_at=self.clock()))
        if not receipt.queued:
            return receipt
        return self.service.submit(TurnRevised(**self._header(), request_source=dependency,
            conversation=ConversationView(turn_id=record.id, turn_revision=record.revision)))


def selected_memory_entry(source: SourceVersion, text: str, *, binding: GuardedPayloadBinding,
                          received_at: datetime, expires_at: datetime) -> SelectedMemoryEntry:
    scope = binding.scope
    dependency = memory_dependency(source)
    if (source.owner_id, source.area_id) != (scope.owner_id, scope.area_id):
        raise ValueError("Memory evidence belongs to another scope")
    if binding.dependencies != (dependency,):
        raise ValueError("Memory entry must retain the exact selected source")
    if expires_at <= received_at:
        raise ValueError("Selected memory is already expired")
    return SelectedMemoryEntry(entry_id=f"memory:{source.source_id}", scope=scope,
        summary=text, sources=binding.dependencies, guard_binding=binding,
        evidence_kind="canonical_record", received_at=received_at,
        observed_at=source.source_time, fresh_until=expires_at, expires_at=expires_at)


def working_observation_entry(state: SessionWorkingMemoryState, observation: VisualObservation, *,
                              binding: GuardedPayloadBinding, frames: tuple,
                              received_at: datetime, fresh_seconds: float = 10.0) -> ObservationEntry | None:
    scope = binding.scope
    if (state.owner_id, state.session_id) != (scope.owner_id, scope.session_id) or (
            observation.owner_id, observation.session_id) != (scope.owner_id, scope.session_id):
        raise ValueError("Working evidence belongs to another session")
    if state.last_observation_id != observation.id or state.last_observation_ended_at != observation.ended_at:
        raise ValueError("Observation does not match the canonical working-memory watermark")
    if not 0 < fresh_seconds <= 60:
        raise ValueError("Visual freshness must be positive and bounded")
    deadline = min(state.expires_at, observation.ended_at + timedelta(seconds=fresh_seconds))
    if deadline <= received_at:
        return None
    if not binding.dependencies:
        raise ValueError("Visual evidence needs canonical source dependencies")
    if not frames or observation.primary_visual_frame_id not in {frame.frame_id for frame in frames}:
        raise ValueError("Visual evidence needs its actual captured frame references")
    if any(frame.captured_at > observation.ended_at for frame in frames):
        raise ValueError("A later camera frame cannot support an earlier observation")
    # The original summary remains intact. Add only complete OCR items that fit;
    # never truncate a number or instruction into a different asserted value.
    text = observation.summary
    for item in observation.visible_text:
        line = f"\nVisible text (confidence {item.confidence:.2f}): {item.text}"
        if len(text) + len(line) <= 2000:
            text += line
    return ObservationEntry(entry_id="working_observation", scope=scope,
        summary=text, frames=frames, confidence=observation.confidence,
        sources=binding.dependencies, guard_binding=binding, evidence_kind="observed",
        received_at=received_at, observed_at=observation.ended_at,
        fresh_until=deadline, expires_at=deadline)
