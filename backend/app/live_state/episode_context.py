"""Bounded projection of the EXISTING responsibility episode owner.

Frozen responsibility/check/episode references retain their canonical identities.
Cross-session source originals are usable only through exact EpisodeCheckOwner
proof; ordinary context_event dependencies remain session scoped. This module
does not authorize admission, infer a duty, settle evidence or send notifications.
"""
from __future__ import annotations

from datetime import timedelta

from app.knowledge.models import SourceError
from app.live_state.models import AggregateRef, CapabilitiesChanged, CheckEntry
from app.live_state.producers import CanonicalStateProducer
from app.live_state.refresh import CanonicalRefreshWorker
from app.live_state.episode_relevance import (
    activity_label, activity_original, current_activity_entry, read_current_activity, relevance_for_check,
)
from app.reminders.episode_checks import EpisodeCheckOwner, EpisodeCheckRead
from app.live_state.perception import DurableResponsibilityCapability
from app.runtime.events import DomainEventName
from app.tools.inputs import ToolInputError


EPISODE_NAMESPACES = frozenset(("responsibility", "responsibility_check", "episode"))


def episode_references(snapshot):
    return tuple(ref for ref in snapshot.dependencies if ref.source_namespace in EPISODE_NAMESPACES)


def service_episode_owner(services):
    """Actual service-owned initialized instance, never a feature-flag promise."""
    attention = getattr(services, "attention_service", None)
    runtime = getattr(attention, "responsibilities", None)
    owner = getattr(runtime, "episode_checks", None)
    if (getattr(services, "_closed", False) or getattr(attention, "_closed", False)
            or getattr(runtime, "_closed", False) or not isinstance(owner, EpisodeCheckOwner)
            or owner.capability.status != "available"):
        return None
    return owner


async def validate_episode_dependencies(canonical, connection, dependencies):
    """Recheck exact relationships under the real owner guard in this transaction.

    The small relation read is just a lookup hint. Only owner.guard validates and
    yields authority. An unpaired namespace or invented relationship is rejected.
    Its locks stay held until the caller's actual provider/effect write commits.
    """
    refs = tuple(ref for ref in dependencies if ref.source_namespace in EPISODE_NAMESPACES)
    if not refs:
        return ()
    adapter = getattr(canonical, "episode_context", None)
    if adapter is None:
        raise SourceError("live_episode_owner_unavailable", "No canonical episode owner is attached.")
    owner = adapter.require_owner()
    if any(ref.generation is not None for ref in refs):
        raise SourceError("live_episode_reference_invalid", "Episode owners do not define lifecycle generations.")
    by_key = {(ref.source_namespace, ref.source_id): ref for ref in refs}
    if len(by_key) != len(refs):
        raise SourceError("live_episode_reference_invalid", "Canonical episode references must be unambiguous.")
    checks = sorted((ref for ref in refs if ref.source_namespace == "responsibility_check"), key=lambda r: r.source_id)
    if not checks or len(checks) > adapter.limit:
        raise SourceError("live_episode_reference_invalid", "Episode proof needs a bounded check set.")
    covered, snapshots = set(), []
    try:
        for check in checks:
            relation = await connection.fetchrow("""select responsibility_id,episode_id
                from jarvis_responsibility_occurrences where owner_id=$1 and id=$2""",
                canonical.scope.owner_id, check.source_id)
            responsibility = by_key.get(("responsibility", relation["responsibility_id"])) if relation else None
            episode = by_key.get(("episode", relation["episode_id"])) if relation else None
            if responsibility is None or episode is None:
                raise SourceError("live_episode_reference_invalid", "The check needs its exact canonical rule and episode.")
            async with owner.guard(canonical.scope, responsibility=responsibility, check=check,
                                   episode=episode, connection=connection) as (borrowed, snapshot):
                if borrowed is not connection or set(episode_references(snapshot)) != {responsibility, check, episode}:
                    raise SourceError("live_episode_reference_invalid", "The owner proof does not match this assignment.")
                snapshots.append(snapshot)
                covered.update((responsibility, check, episode))
    except ToolInputError as error:
        raise SourceError("live_episode_" + error.code, "The canonical episode check is no longer applicable.") from error
    if covered != set(refs):
        raise SourceError("live_episode_reference_invalid", "Unpaired owner or episode references are not proof.")
    return tuple(snapshots)


async def link_episode_sources(canonical, connection, root, snapshots):
    """Link correction edges for explicitly owner-proven historical requests.

    No generic foreign context-event exception is introduced. This function is
    called only with snapshots just yielded by the canonical episode guard.
    """
    dependencies = {}
    roots = set()
    for snapshot in snapshots:
        roots.update(snapshot.source_payload_ids)
        for ref in snapshot.dependencies:
            if ref.source_namespace == "context_event":
                previous = dependencies.setdefault(ref.source_id, ref)
                if previous != ref:
                    raise SourceError("live_episode_source_changed", "Conflicting episode originals.")
    if not dependencies:
        return
    records = await canonical.context._records(connection, canonical.scope.owner_id, list(dependencies))
    rows = []
    for identity, expected in dependencies.items():
        record = records.get(identity)
        if (record is None or record.owner_id != canonical.scope.owner_id or record.revision != expected.revision
                or record.availability != "available" or record.content is None or not record.is_current
                or expected.generation is not None or record.payload_id not in roots):
            raise SourceError("live_episode_source_changed", "Original episode evidence changed.")
        rows.append((canonical.scope.owner_id, root, record.id, record.revision, record.payload_id, "current"))
    await connection.executemany("""insert into jarvis_context_event_dependencies
        (owner_id,dependent_payload_id,event_id,event_revision,source_payload_id,source_usage)
        values($1,$2,$3,$4,$5,$6) on conflict do nothing""", rows)


class EpisodeContextRuntime:
    """One optional background producer; snapshot caches are bounded and disposable."""

    def __init__(self, *, service, canonical, owner, bus, clock, owner_current=None,
                 unavailable_status="unavailable", interval_seconds=1.0, timeout_seconds=2.0,
                 ttl_seconds=10.0, limit=8):
        if type(limit) is not int or not 1 <= limit <= 8 or unavailable_status not in {"disabled", "unavailable"}:
            raise ValueError("Invalid episode context bounds or unavailable status")
        self.service, self.canonical, self.owner, self.clock = service, canonical, owner, clock
        self.owner_current = owner_current
        self.unavailable_status, self.ttl_seconds, self.limit = unavailable_status, ttl_seconds, limit
        self.producer = CanonicalStateProducer(service, producer_id="canonical-episodes", clock=clock)
        self.worker = CanonicalRefreshWorker(scope=canonical.scope, bus=bus,
            event_names=(DomainEventName.MEMORY_ITEM_CHANGED, DomainEventName.OWNER_WORKING_MEMORY_UPDATED,
                         DomainEventName.WORKING_MEMORY_UPDATED),
            read=self._read, apply=self._apply, invalidate=self._invalidate,
            interval_seconds=interval_seconds, timeout_seconds=timeout_seconds)
        self._cache, self._published = {}, {}
        self._closed = False
        self.omissions = ()
        self.status = unavailable_status

    def require_owner(self):
        if (self._closed or self.owner is None or self.owner.capability.status != "available"
                or (self.owner_current is not None and self.owner_current() is not self.owner)):
            raise SourceError("live_episode_owner_unavailable", "The initialized episode owner is unavailable or replaced.")
        return self.owner

    def _capability(self, status):
        self.status = status
        previous = self.service.snapshot().capabilities
        if previous.durable_responsibilities == status:
            return
        receipt = self.service.submit(CapabilitiesChanged(**self.producer._header(),
            capabilities=previous.model_copy(update={"durable_responsibilities": status})))
        if not receipt.queued:
            raise SourceError("live_episode_projection_unavailable", receipt.reason)

    def start(self):
        self.producer.start()
        self._capability(self.unavailable_status)
        if self.owner is not None:
            self.worker.start()

    async def _read(self):
        try:
            owner = self.require_owner()
        except SourceError:
            return EpisodeCheckRead(capability=DurableResponsibilityCapability(
                owner_namespace="responsibility", status=self.unavailable_status))
        return await owner.read(self.canonical.scope, limit=self.limit)

    async def _apply(self, read):
        entries, cache, published, omissions = [], {}, {}, list(read.gaps)
        now = self.clock()
        deadline = now + timedelta(seconds=self.ttl_seconds)
        if read.capability.status == "available":
            self.require_owner()
            activity = None
            if any(item.check.activation_cues for item in read.snapshots[:self.limit]):
                try:
                    async with self.canonical._transaction() as connection:
                        activity = await read_current_activity(self.canonical, connection,
                            self.service.snapshot(), now=now)
                except SourceError as error:
                    omissions.append(error.code)
            for candidate in read.snapshots[:self.limit]:
                dependencies = episode_references(candidate)
                identity = f"episode-check:{candidate.check.id}"
                try:
                    async with self.canonical._transaction() as connection:
                        fresh, = await validate_episode_dependencies(self.canonical, connection, dependencies)
                        if fresh != candidate:
                            raise SourceError("live_episode_source_changed", "The owner changed during projection.")
                        text = ("Unresolved episode check, not proof of action completion: " + fresh.check.question)
                        previous = self._cache.get(identity)
                        if previous is not None and previous[0] == fresh:
                            binding = previous[1]
                            await self.canonical._lock_roots(connection, binding.payload_root_ids)
                        else:
                            await self.canonical._lock_roots(connection, fresh.source_payload_ids)
                            binding = await self.canonical._materialize(connection, text, dependencies, fresh.source_payload_ids)
                            await link_episode_sources(self.canonical, connection, binding.payload_root_ids[0], (fresh,))
                        if deadline <= self.clock():
                            raise SourceError("live_episode_context_expired", "The episode projection expired before publication.")
                        check = fresh.check
                        entries.append(CheckEntry(entry_id=identity, scope=self.canonical.scope,
                            check_id=check.id, check_revision=check.revision, lifecycle=check.lifecycle.value,
                            canonical_owner=AggregateRef(namespace="responsibility", aggregate_id=check.canonical_owner_ref.source_id,
                                revision=check.canonical_owner_ref.revision), durable=True, question=text,
                            sources=dependencies, guard_binding=binding, evidence_kind="canonical_record",
                            received_at=now, observed_at=fresh.payload.episode_started_at,
                            fresh_until=deadline, expires_at=deadline))
                        cache[identity] = (fresh, binding)
                        relevance = relevance_for_check(check, fresh.payload.episode_started_at,
                            activity, now=self.clock())
                        if relevance.relevant:
                            proof = relevance.activity
                            assignment_dependencies = dependencies + ((proof.source,) if proof else ())
                            published[identity] = (check.model_copy(update={
                                "relevance_now": True, "dependency_versions": assignment_dependencies}), binding, proof)
                        else:
                            omissions.append("episode_activation_" + relevance.reason + ":" + str(check.id))
                except SourceError as error:
                    omissions.append(error.code + ":" + str(candidate.check.id))
        if self._closed:
            return
        for identity in self._cache.keys() - cache.keys():
            receipt = self.producer.retract(identity, "superseded")
            if not receipt.queued:
                raise SourceError("live_episode_projection_unavailable", receipt.reason)
        for entry in entries:
            receipt = self.producer.publish(entry)
            if not receipt.queued:
                raise SourceError("live_episode_projection_unavailable", receipt.reason)
        self._capability(read.capability.status)
        await self.service.wait_idle()
        self._cache, self._published, self.omissions = cache, published, tuple(omissions)

    def checks(self, snapshot):
        try:
            self.require_owner()
        except SourceError:
            return ()
        if snapshot.capabilities.durable_responsibilities != "available":
            return ()
        entries = {entry.entry_id:entry for entry in snapshot.entries}
        now, result = self.clock(), []
        activity_entry = current_activity_entry(snapshot, now=now)
        for identity, (check, binding, proof) in self._published.items():
            entry = entries.get(identity)
            if (entry is not None and entry.validity == "active" and entry.guard_binding == binding
                    and entry.check_revision == check.revision
                    and (proof is None or (now < proof.expires_at and activity_entry is not None
                         and activity_entry.sources == (proof.source,)))
                    and all(t is None or t > now for t in (entry.fresh_until, entry.expires_at))):
                result.append(check)
        return tuple(result)

    async def validate_assignment(self, connection, brief, episodes):
        """Narrow binder seam: recheck cue relevance at bind AND actual effects.

        CanonicalPerceptionBinder must call this after its owner-version proof.
        The fixed assignment keeps its original activity source and expiry. A
        newer same-activity observation need not invalidate it; a changed or
        uncertain activity does. This cannot resolve, cancel or admit a check.
        """
        assigned = {question.check_ref.source_id for question in brief.questions
                    if question.check_ref is not None
                    and question.check_ref.source_namespace == "responsibility_check"}
        conditional = tuple(item for item in episodes if item.check.id in assigned and item.check.activation_cues)
        if not conditional:
            return
        now = self.clock()
        current = await read_current_activity(self.canonical, connection, self.service.snapshot(), now=now)
        originals = []
        for source in brief.dependencies:
            if source.source_namespace == "context_event":
                activity = await activity_original(self.canonical, connection, source, now=now)
                if activity is not None:
                    originals.append(activity)
        for item in conditional:
            current_relevance = relevance_for_check(item.check, item.payload.episode_started_at, current, now=now)
            valid_original = any(
                current is not None and activity_label(original.label) == activity_label(current.label)
                and relevance_for_check(item.check, item.payload.episode_started_at, original, now=now).relevant
                for original in originals)
            if not current_relevance.relevant or not valid_original:
                raise SourceError("live_episode_cue_inapplicable", "The assignment's original activity cue is no longer applicable.")

    async def _invalidate(self, reason):
        if not self._closed:
            await self._apply(EpisodeCheckRead(capability=DurableResponsibilityCapability(
                owner_namespace="responsibility", status="unavailable"), gaps=(str(reason),)))

    async def close(self):
        self._closed = True
        await self.worker.close()
        self._cache.clear()
        self._published.clear()
