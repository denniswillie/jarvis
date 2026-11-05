"""Disposable context-relevant selection through the existing memory authority.

The context reader and single lookup worker are independent of conversation and
minimal perception readiness. Neither admits work/actions or creates memories.
Provider prose is never published: returned references select canonical originals.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import timedelta
import json
import math
import time
from uuid import UUID

from app.knowledge.models import SourceError
from app.live_state.canonical import context_dependency, memory_dependency
from app.live_state.episode_relevance import current_activity_entry, read_current_activity, activity_label
from app.live_state.models import CapabilitiesChanged, GuardedPayloadBinding, SelectedMemoryEntry
from app.live_state.perception import PerceptionActivity
from app.live_state.producers import CanonicalStateProducer
from app.live_state.projection import VoiceSelection
from app.live_state.refresh import CanonicalRefreshWorker
from app.runtime.events import DomainEventName


PREFIX = "historical:"
CANDIDATE_LIMIT = 8
MEMORY_TEXT_LIMIT = 950
OMISSION_DIAGNOSTIC_LIMIT = 32


def _normalized(value):
    return " ".join(value.casefold().split())


def _dependencies(values):
    result = {}
    for source in values:
        key = source.source_namespace, source.source_id
        if key in result and result[key] != source:
            raise SourceError("historical_context_conflict", "Context source versions disagree.")
        result[key] = source
    return tuple(result.values())


@dataclass(frozen=True)
class HistoricalContext:
    fingerprint: tuple
    semantic_key: tuple
    query: str
    dependencies: tuple
    roots: tuple
    expires_at: object
    intent_ref: object | None = None
    activity: PerceptionActivity | None = None


class HistoricalEnrichment:
    def __init__(self, *, service, canonical, retrieval, bus, clock,
                 interval_seconds=1.0, timeout_seconds=2.0, ttl_seconds=10.0,
                 max_sources=4, deadline_seconds=5.0, calls_per_minute=2,
                 debounce_seconds=0.15, monotonic=time.monotonic,
                 publication_deadline_seconds=5.0):
        if (type(max_sources) is not int or not 1 <= max_sources <= 8
                or type(calls_per_minute) is not int or not 1 <= calls_per_minute <= 2
                or not math.isfinite(deadline_seconds) or not 0 < deadline_seconds <= 5
                or not math.isfinite(publication_deadline_seconds) or not 0 < publication_deadline_seconds <= 5
                or not math.isfinite(debounce_seconds) or not 0 <= debounce_seconds <= 1
                or not math.isfinite(ttl_seconds) or not 0 < ttl_seconds <= 60):
            raise ValueError("Invalid historical enrichment bounds")
        self.service, self.canonical, self.retrieval = service, canonical, retrieval
        self.clock, self.monotonic = clock, monotonic
        self.scope, self.epoch = canonical.scope, service.snapshot().cursor.epoch
        self.max_sources, self.deadline_seconds = max_sources, deadline_seconds
        self.publication_deadline_seconds = publication_deadline_seconds
        self.calls_per_minute, self.debounce_seconds = calls_per_minute, debounce_seconds
        self.ttl_seconds = ttl_seconds
        self.producer = CanonicalStateProducer(service, producer_id="historical-enrichment", clock=clock)
        self.worker = CanonicalRefreshWorker(scope=self.scope, bus=bus,
            event_names=(DomainEventName.MEMORY_ITEM_CHANGED, DomainEventName.WORKING_MEMORY_UPDATED,
                         DomainEventName.OWNER_WORKING_MEMORY_UPDATED),
            read=self._read_context, apply=self._observe, invalidate=self._invalidate,
            interval_seconds=interval_seconds, timeout_seconds=timeout_seconds)
        self._wanted = self._pending = None
        self._generation = 0
        self._not_before = 0.0
        self._selected = ()
        # Retrieval rank is preserved. The reserve is not published state and
        # cannot authorize use without the same original/version/effect guards.
        self._candidates = ()
        self._candidate_omissions = deque(maxlen=OMISSION_DIAGNOSTIC_LIMIT)
        self._bindings, self._published = {}, set()
        self._calls, self._wake = deque(), asyncio.Event()
        self._publication = asyncio.Lock()
        self._publication_wake = asyncio.Event()
        self._publication_task = None
        self._selection_ready = False
        self._task = None
        self._closed = False
        self.status, self.last_error = "warming", None
        self.lookup_count = self.stale_results = self.omitted_sources = 0
        self.publication_attempts = self.publication_completed = self.publication_failures = 0
        self.last_publication_ms = None

    def _omit_candidate(self, reason, *, phase, rank, source=None, candidate=None):
        """Only typed identity/version fields, never provider text or errors."""
        identity = getattr(source, "source_id", None)
        revision = getattr(source, "revision", None)
        generation = getattr(source, "generation", None)
        if source is None and isinstance(candidate, dict):
            identity = candidate.get("source_id")
            revision, generation = candidate.get("revision"), candidate.get("generation")
        if not isinstance(identity, UUID):
            try:
                identity = UUID(identity) if isinstance(identity, str) and len(identity) <= 36 else None
            except ValueError:
                identity = None
        record = {"lookup": self.lookup_count, "context_generation": self._generation,
            "phase": phase, "candidate_rank": rank, "reason": reason,
            "source_id": str(identity) if identity is not None else None,
            "revision": revision if type(revision) is int and revision > 0 else None,
            "generation": generation if type(generation) is int and generation > 0 else None}
        self.omitted_sources += 1
        for previous in self._candidate_omissions:
            if all(previous[key] == value for key, value in record.items()):
                previous["occurrences"] += 1
                return
        self._candidate_omissions.append(record | {"occurrences": 1})

    def candidate_diagnostics(self):
        return {"candidate_limit": CANDIDATE_LIMIT, "selected_limit": self.max_sources,
            "memory_text_limit": MEMORY_TEXT_LIMIT, "retained_candidates": len(self._candidates),
            "published_count": len(self._published), "omission_events": self.omitted_sources,
            "recent_omissions": [dict(record) for record in self._candidate_omissions],
            "record_limit": OMISSION_DIAGNOSTIC_LIMIT, "raw_content_included": False}

    def start(self):
        if self._task is not None or self._closed:
            raise RuntimeError("Historical enrichment cannot restart")
        self.producer.start()
        self.worker.start()
        self._task = asyncio.create_task(self._run(), name="live-historical-enrichment")
        self._publication_task = asyncio.create_task(self._run_publication(), name="live-historical-publication")

    def _inputs(self, snapshot):
        if (snapshot.scope != self.scope or snapshot.cursor.epoch != self.epoch
                or snapshot.readiness != "ready" or self._closed):
            return (), ()
        now = self.clock()
        entries = [entry for entry in snapshot.entries if entry.validity == "active"
            and entry.guard_binding is not None and entry.scope == self.scope
            and all(t is None or t > now for t in (entry.fresh_until, entry.expires_at))]
        intent = next((entry for entry in entries if entry.entry_id == "conversation:request"
            and entry.kind == "focus" and len(entry.sources) == 1
            and entry.sources[0].source_namespace == "context_event"), None)
        activity = current_activity_entry(snapshot, now=now)
        checks = sorted((entry for entry in entries if entry.kind == "situational_check"
            and entry.lifecycle in {"candidate", "tracking"}), key=lambda entry: entry.entry_id)[:3]
        selected = tuple(entry for entry in (intent, activity, *checks) if entry is not None)
        # Projection renewal, unrelated tasks and our own publications are not
        # new intent. Original source/frame versions remain an immediate fence.
        fingerprint = tuple((entry.entry_id, entry.sources, entry.guard_binding.payload_root_ids,
            getattr(entry, "goal", None), getattr(entry, "question", None), entry.observed_at,
            getattr(entry, "frames", ())) for entry in selected)
        return selected, fingerprint

    async def _read_context(self):
        snapshot = self.service.snapshot()
        entries, fingerprint = self._inputs(snapshot)
        if not entries:
            return None
        parts, dependencies, roots, deadlines, keys = [], [], [], [], []
        intent_ref = activity_view = None
        async with self.canonical._transaction() as connection:
            for entry in entries:
                await self.canonical._lock_roots(connection, entry.guard_binding.payload_root_ids)
                await self.canonical._validate_dependencies(connection, entry.sources)
                if entry.kind == "focus":
                    record = await self.canonical.context._record(connection, self.scope.owner_id,
                                                                   entry.sources[0].source_id)
                    text = (record.content or {}).get("text")
                    if (not record.authorizes_request or context_dependency(record) != entry.sources[0]
                            or not isinstance(text, str) or not text.strip() or len(text) > 800):
                        continue
                    line = "Current request: " + text
                    keys.append(("intent", _normalized(text)))
                    intent_ref = entry.sources[0]
                elif entry.kind == "observation":
                    activity = await read_current_activity(self.canonical, connection, snapshot, now=self.clock())
                    if activity is None or activity.confidence < .7 or activity_label(activity.label) is None:
                        continue
                    line = "Current observed activity: " + activity.label
                    keys.append(("activity", activity_label(activity.label)))
                    activity_view = PerceptionActivity(label=activity.label, confidence=activity.confidence,
                        uncertainty="Analyzer interpretation, not proof of a completed action.")
                    deadlines.append(activity.expires_at)
                else:
                    if len(entry.question) > 300:
                        continue
                    line = "Unresolved check: " + entry.question
                    keys.append(("check", entry.check_id, entry.check_revision, _normalized(entry.question)))
                parts.append(line)
                dependencies.extend(entry.sources)
                roots.extend(entry.guard_binding.payload_root_ids)
                deadlines.extend(t for t in (entry.fresh_until, entry.expires_at) if t is not None)
        if not parts:
            return None
        # Bounded complete statements, not a model that guesses the next task.
        query = "Relevant historical facts and preferences for this context:\n" + "\n".join(parts)
        if len(query) > 2000:
            raise SourceError("historical_query_budget", "Context exceeds the retrieval budget.")
        return HistoricalContext(fingerprint, tuple(keys), query, _dependencies(dependencies),
            tuple(dict.fromkeys(roots)), min(deadlines, default=self.clock() + timedelta(seconds=self.ttl_seconds)),
            intent_ref, activity_view)

    def current_context(self, snapshot=None):
        context = self._wanted
        if (context is None or self._closed or self.clock() >= context.expires_at
                or self._inputs(snapshot or self.service.snapshot())[1] != context.fingerprint):
            return None
        return context

    def _retract(self, reason):
        for identity in self._published:
            self.producer.retract(identity, reason)
        self._published.clear()
        self._bindings.clear()

    async def _capability(self):
        await self.service.wait_idle()
        available = "available" if self.status == "available" else "unavailable"
        capabilities = self.service.snapshot().capabilities
        if capabilities.historical_retrieval != available:
            self.service.submit(CapabilitiesChanged(**self.producer._header(),
                capabilities=capabilities.model_copy(update={"historical_retrieval": available})))

    async def _observe(self, context):
        if self._closed:
            return
        if context is not None and self._inputs(self.service.snapshot())[1] != context.fingerprint:
            return  # Reader lost a relevant-source race; polling reads the newest inputs.
        changed = (context.semantic_key if context else None) != (
            self._wanted.semantic_key if self._wanted else None)
        replaced = (context.fingerprint if context else None) != (
            self._wanted.fingerprint if self._wanted else None)
        self._wanted = context
        if changed:
            self._generation += 1
            self._selected = ()
            self._candidates = ()
            self._selection_ready = False
            self._retract("superseded")
            self._pending = self._generation if context is not None else None
            self._not_before = self.monotonic() + self.debounce_seconds
            self.status = "queued" if context else "waiting_for_context"
            self._wake.set()
        elif replaced:
            # Equivalent wording can still refer to a replacement request or
            # activity frame. Retire the old intent binding, not the memory.
            self._retract("superseded")
        if self._selection_ready and context is not None:
            self._publication_wake.set()
        await self._capability()

    async def _invalidate(self, reason):
        if self._closed:
            return
        self._generation += 1
        self._wanted = self._pending = None
        self._selected = ()
        self._candidates = ()
        self._selection_ready = False
        self._retract("source_withdrawn")
        self.status, self.last_error = "unavailable", str(reason)
        self._wake.set()
        await self._capability()

    async def _lookup(self, context, generation):
        result = await self.retrieval.recall(self.scope.owner_id, self.scope.area_id, context.query)
        if generation != self._generation or self._closed:
            self.stale_results += 1
            return None
        if result.get("status") != "available":
            raise SourceError("historical_retrieval_unavailable", "Historical retrieval is unavailable.")
        selected, seen = [], set()
        for rank, candidate in enumerate(result.get("sources", ())[:CANDIDATE_LIMIT], start=1):
            if not isinstance(candidate, dict) or candidate.get("content_included") is not True:
                self._omit_candidate("content_not_included", phase="lookup", rank=rank, candidate=candidate)
                continue
            try:
                source = await self.canonical.sources.resolve_document(
                    self.scope.owner_id, self.scope.area_id, candidate["document_id"])
                original = source.original_evidence()
                if (not source.is_current or (source.owner_id, source.area_id) != (
                        self.scope.owner_id, self.scope.area_id)
                        or any(candidate.get(key) != original[key] for key in (
                            "owner_id", "area_id", "source_id", "revision", "generation", "payload_id",
                            "document_id", "content_digest", "content"))):
                    self._omit_candidate("source_version_mismatch", phase="lookup", rank=rank, source=source)
                    continue
                if source.source_id in seen:
                    self._omit_candidate("duplicate_source", phase="lookup", rank=rank, source=source)
                    continue
                selected.append(source)
                seen.add(source.source_id)
            except SourceError:
                self._omit_candidate("canonical_source_unavailable", phase="lookup", rank=rank, candidate=candidate)
            except KeyError:
                self._omit_candidate("reference_missing", phase="lookup", rank=rank, candidate=candidate)
        return tuple(selected)

    def _publication_current(self, context, generation):
        current = self.current_context()
        return (self._selection_ready and current is not None
                and generation == self._generation and context.expires_at > self.clock()
                and current.fingerprint == context.fingerprint
                and current.dependencies == context.dependencies and current.roots == context.roots)

    async def _publish(self):
        async with self._publication:
            context = self.current_context()
            if context is None:
                return False
            generation = self._generation
            entries, cache, retained = [], {}, []
            for rank, selected in enumerate(self._candidates, start=1):
                if len(entries) >= self.max_sources:
                    self._omit_candidate("selection_capacity", phase="publication", rank=rank, source=selected)
                    continue
                try:
                    source = await self.canonical.sources.revalidate(selected)
                    if not source.is_current:
                        raise SourceError("historical_source_changed", "Original is no longer current.")
                    text = (f"Historical source ({source.trust}; {source.time_basis}), not a current observation: "
                            + json.dumps(source.content, ensure_ascii=True, sort_keys=True))
                    if len(text) > MEMORY_TEXT_LIMIT:
                        self._omit_candidate("text_budget_exceeded", phase="publication", rank=rank, source=source)
                        continue  # Never shorten code, numbers or negation into a different fact.
                    identity = PREFIX + str(source.source_id)
                    proof = (source, context.dependencies, context.roots, text)
                    previous = self._bindings.get(identity)
                    if previous is not None and previous[0] == proof and await self.canonical.available(previous[1]):
                        binding = previous[1]
                    else:
                        original = await self.canonical.bind_memory(source, text)
                        dependencies = _dependencies((*context.dependencies, memory_dependency(source)))
                        binding = await self.canonical(VoiceSelection(self.scope, self.service.snapshot().cursor,
                            text, dependencies, (*context.roots, *original.payload_root_ids), (), (), context.expires_at))
                    if not self._publication_current(context, generation):
                        return False
                    expires = min(context.expires_at, self.clock() + timedelta(seconds=self.ttl_seconds))
                    entries.append(SelectedMemoryEntry(entry_id=identity, scope=self.scope,
                        summary=text, sources=binding.dependencies, guard_binding=binding,
                        evidence_kind="canonical_record", received_at=self.clock(), observed_at=source.source_time,
                        fresh_until=expires, expires_at=expires))
                    cache[identity] = (proof, binding)
                    retained.append(source)
                except SourceError:
                    self._omit_candidate("canonical_source_or_binding_unavailable",
                        phase="publication", rank=rank, source=selected)
            if not self._publication_current(context, generation):
                return False
            identities = {entry.entry_id for entry in entries}
            # Validate the existing individual bindings together; this union is
            # an effect guard, NOT a materialized voice/context copy. A source
            # may have changed while a later source was being bound. Enqueue all
            # changes without awaiting while that canonical authority is held.
            guard = GuardedPayloadBinding(scope=self.scope,
                payload_root_ids=tuple(dict.fromkeys((*context.roots,
                    *(root for entry in entries for root in entry.guard_binding.payload_root_ids)))),
                dependencies=_dependencies((*context.dependencies,
                    *(source for entry in entries for source in entry.sources))))
            async with self.canonical.guard(guard):
                if not self._publication_current(context, generation):
                    return False
                for identity in self._published - identities:
                    self.producer.retract(identity, "source_withdrawn")
                for entry in entries:
                    if not self.producer.publish(entry).queued:
                        raise SourceError("historical_projection_unavailable", "State writer rejected enrichment.")
                self._published, self._bindings, self._selected = identities, cache, tuple(retained)
            await self.service.wait_idle()
            return True

    async def _run_publication(self):
        """One coalesced, bounded I/O stage, never part of the refresh deadline.

        Context polling keeps observing replacements while publication runs.
        TTL-only renewals do not discard valid work. Failure withdraws the view
        but retains candidates for a guarded publication retry on the next poll,
        rather than spending another retrieval call on unchanged context.
        """
        while not self._closed:
            await self._publication_wake.wait()
            self._publication_wake.clear()
            context, generation = self.current_context(), self._generation
            if context is None or not self._selection_ready:
                continue
            self.publication_attempts += 1
            started = time.monotonic()
            try:
                async with asyncio.timeout(self.publication_deadline_seconds):
                    published = await self._publish()
                if published and self._publication_current(context, generation):
                    self.publication_completed += 1
                    self.status, self.last_error = "available", None
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.publication_failures += 1
                if self._publication_current(context, generation):
                    self._retract("source_withdrawn")
                    self.status = "unavailable"
                    self.last_error = ("historical_publication_timeout" if isinstance(error, TimeoutError)
                                       else getattr(error, "code", type(error).__name__))
                    # Coalesce polls received during the failed attempt. A new
                    # canonical poll may retry, with unchanged global budgets.
                    self._publication_wake.clear()
            finally:
                self.last_publication_ms = round((time.monotonic() - started) * 1000, 3)
            if not self._closed:
                await self._capability()

    async def _run(self):
        while not self._closed:
            await self._wake.wait()
            self._wake.clear()
            while self._pending is not None and not self._closed:
                now = self.monotonic()
                while self._calls and self._calls[0] <= now - 60:
                    self._calls.popleft()
                delay = max(0, self._not_before - now,
                    self._calls[0] + 60 - now if len(self._calls) >= self.calls_per_minute else 0)
                if delay:
                    self.status = "budget_wait" if len(self._calls) >= self.calls_per_minute else "coalescing"
                    try:
                        await asyncio.wait_for(self._wake.wait(), delay)
                        self._wake.clear()
                    except TimeoutError:
                        pass
                    continue
                generation, context = self._pending, self._wanted
                self._pending = None
                if context is None:
                    continue
                self._calls.append(now)
                self.lookup_count += 1
                self.status, self.last_error = "running", None
                try:
                    async with asyncio.timeout(self.deadline_seconds):
                        selected = await self._lookup(context, generation)
                        if selected is not None and generation == self._generation and not self._closed:
                            self._candidates = selected
                            self._selected = selected[:self.max_sources]
                            self._selection_ready = True
                            self.status = "publishing"
                            self._publication_wake.set()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    if generation == self._generation:
                        self._selected = ()
                        self._candidates = ()
                        self._selection_ready = False
                        self._retract("source_withdrawn")
                        self.status = "unavailable"
                        self.last_error = getattr(error, "code", type(error).__name__)
                if not self._closed:
                    await self._capability()

    async def close(self):
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        self._pending = self._wanted = None
        self._selection_ready = False
        owned = [task for task in (self._task, self._publication_task) if task is not None]
        for task in owned:
            task.cancel()
        try:
            await self.worker.close()
        finally:
            if owned:
                await asyncio.gather(*owned, return_exceptions=True)
            self._selected = ()
            self._candidates = ()
            self._retract("cancelled")
            self.status = "closed"
