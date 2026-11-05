"""Project the existing working-memory authority into Live State asynchronously.

This reads canonical observation originals, not retained operational prose. It
does not create another visual analyzer, memory store or responsibility owner.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from app.images import ImageReference
from app.knowledge.models import SourceError
from app.knowledge.observations import OBSERVATION_NAMESPACE
from app.live_state.canonical import CanonicalContextBinder, context_dependency
from app.live_state.models import FrameRef, GuardedPayloadBinding
from app.live_state.producers import CanonicalStateProducer, working_observation_entry
from app.live_state.refresh import CanonicalRefreshWorker
from app.perception.repository import _observation_from_row
from app.runtime.events import DomainEventName


def observation_from_record(record):
    if (record.source_namespace != OBSERVATION_NAMESPACE or record.kind != "observation"
            or record.availability != "available" or not record.is_current or record.content is None
            or record.content.get("origin") != "perception_observations"):
        raise SourceError("live_state_observation_unavailable", "Current canonical visual evidence is unavailable.")
    raw = dict(record.content["record"])
    for field in ("id", "session_id", "primary_visual_frame_id"):
        if raw.get(field) is not None:
            raw[field] = UUID(str(raw[field]))
    for field in ("started_at", "ended_at", "compacted_at", "created_at"):
        if isinstance(raw.get(field), str):
            raw[field] = datetime.fromisoformat(raw[field])
    observation = _observation_from_row(raw)
    if ((observation.owner_id, observation.session_id) != (record.owner_id, record.session_id)
            or str(observation.id) != record.source_event_id
            or observation.ended_at != record.source_time):
        raise SourceError("live_state_observation_mismatch", "Visual provenance disagrees with its canonical record.")
    frames = []
    allowed = {str(item["frame_id"]) for item in record.content.get("frame_links", ())}
    for metadata in record.content.get("images", ()):
        image = ImageReference.from_metadata(metadata, owner_id=record.owner_id)
        if (image.session_id != record.session_id or str(image.frame_id) not in allowed
                or image.visual_session_id is None
                or not observation.started_at <= image.captured_at <= observation.ended_at):
            raise SourceError("live_state_frame_provenance", "The visual frame lacks exact camera provenance.")
        frames.append(FrameRef(frame_id=image.frame_id, stream_epoch=image.visual_session_id,
            captured_at=image.captured_at))
    if len(frames) > 4 or len({frame.frame_id for frame in frames}) != len(frames):
        raise SourceError("live_state_frame_budget", "The current visual window exceeds its contract.")
    return observation, tuple(frames)


class PerceptionLiveAdapter:
    def __init__(self, *, service, working_memory, sources, bus, clock,
                 interval_seconds=1.0, timeout_seconds=2.0):
        self.service, self.working_memory, self.clock = service, working_memory, clock
        self.scope = service.snapshot().scope
        self.binder = CanonicalContextBinder(sources, self.scope, clock=clock,
            timeout_seconds=timeout_seconds)
        self.producer = CanonicalStateProducer(service, producer_id="canonical-perception", clock=clock)
        self.worker = CanonicalRefreshWorker(scope=self.scope, bus=bus,
            event_names=(DomainEventName.WORKING_MEMORY_UPDATED,
                         DomainEventName.PERCEPTION_OBSERVATION_COMPLETED),
            read=self._read, apply=self._apply, invalidate=self.invalidate,
            interval_seconds=interval_seconds, timeout_seconds=timeout_seconds)
        self._last_revision = None
        self._last_entry = None
        self.last_invalidation_reason = None

    def start(self):
        self.producer.start()
        self.worker.start()

    async def _read(self):
        state = await self.working_memory.get_session_working_memory(
            owner_id=self.scope.owner_id, session_id=self.scope.session_id)
        if state is None or state.last_observation_id is None or state.expires_at <= self.clock():
            return state, None
        if self._last_revision == state.revision and self._last_entry is not None:
            return state, None
        async with self.binder.sources.pool.acquire() as connection:
            event_id = await connection.fetchval(
                """select e.id from jarvis_observation_sources o
                   join jarvis_context_events e on e.owner_id=o.owner_id and e.payload_id=o.payload_id
                   where o.owner_id=$1 and o.session_id=$2 and o.observation_id=$3
                     and e.session_id=$2 and e.source_namespace=$4
                   order by e.revision desc limit 1""",
                self.scope.owner_id, self.scope.session_id, state.last_observation_id, OBSERVATION_NAMESPACE)
            if event_id is None:
                raise SourceError("live_state_observation_original_missing", "Visual evidence has no canonical original.")
            record = await self.binder.context._record(connection, self.scope.owner_id, event_id)
        return state, record

    async def _apply(self, value):
        state, record = value
        now = self.clock()
        if state is None or state.expires_at <= now or state.last_observation_id is None:
            await self.invalidate("visual_evidence_unavailable")
            return
        if self._last_revision == state.revision and self._last_entry is not None:
            if (self._last_entry.fresh_until <= now
                    or not await self.binder.available(self._last_entry.guard_binding)):
                await self.invalidate("visual_evidence_expired_or_changed")
            return
        observation, frames = observation_from_record(record)
        original = GuardedPayloadBinding(scope=self.scope, payload_root_ids=(record.payload_id,),
            dependencies=(context_dependency(record),))
        entry = working_observation_entry(state, observation, binding=original,
            frames=frames, received_at=now)
        if entry is None:
            await self.invalidate("visual_evidence_expired")
            return
        binding = await self.binder.bind_record(record, entry.summary)
        # A database-backed bind may overlap the next observation. Reconcile
        # against the original authority before publishing a late result.
        current = await self.working_memory.get_session_working_memory(
            owner_id=self.scope.owner_id, session_id=self.scope.session_id)
        if (current is None or current.revision != state.revision
                or current.last_observation_id != state.last_observation_id
                or entry.fresh_until <= self.clock()):
            return
        entry = entry.model_copy(update={"guard_binding": binding})
        receipt = self.producer.publish(entry)
        if not receipt.queued:
            raise SourceError("live_state_projection_unavailable", "The visual projection requires reconciliation.")
        self._last_revision, self._last_entry = state.revision, entry

    async def invalidate(self, reason):
        event_reason = {
            "visual_evidence_expired": "expired",
            "perception_projection_closed": "cancelled",
        }.get(reason, reason if reason in {"superseded", "source_withdrawn", "cancelled", "expired"}
              else "source_withdrawn")
        self.producer.retract("working_observation", event_reason)
        self.last_invalidation_reason = reason
        self._last_revision, self._last_entry = None, None

    async def close(self):
        await self.worker.close()
        await self.invalidate("perception_projection_closed")
