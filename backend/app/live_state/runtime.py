"""Opt-in, disposable Live State composition for one authenticated session.

The session owner calls ``start`` once and awaits ``close`` before its canonical
owners/pool are closed. Construction/start and ``service.snapshot`` perform no
repository or model work. Refresh and binding run in background tasks; ``guard``
belongs at the existing guarded provider-write boundary, never audio ingress.

This module does not admit tasks, select a memory authority, authorize a request,
or recover remote provider context. A fenced runtime or changed area selection
requires the realtime owner to retire remote context and construct a NEW runtime.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import json
import logging
import math
from uuid import UUID

from app.knowledge.models import SourceError
from app.live_state.canonical import CanonicalContextBinder
from app.live_state.episode_context import EpisodeContextRuntime, service_episode_owner
from app.live_state.historical_enrichment import HistoricalEnrichment
from app.live_state.models import AggregateRef, FocusEntry, SessionScope
from app.live_state.perception_adapter import PerceptionLiveAdapter
from app.live_state.perception_runtime import SessionPerceptionRuntime
from app.live_state.producers import CanonicalStateProducer, selected_memory_entry
from app.live_state.refresh import CanonicalRefreshWorker
from app.live_state.service import LiveStateService
from app.live_state.task_context import TaskContextProjector
from app.runtime.events import DomainEventName
from app.runtime.feature_config import MemoryAuthority


_AVAILABILITY_LOGGER = logging.getLogger(__name__)
_AVAILABILITY_LOG_LIMIT = 8


def _utcnow():
    return datetime.now(UTC)


class _SessionBinder(CanonicalContextBinder):
    """Check the actual area-selection owner under the existing source lock."""

    def __init__(self, *args, area_selections, access_check, **kwargs):
        super().__init__(*args, **kwargs)
        self.area_selections, self.access_check = area_selections, access_check
        self.selection = None
        self.scope_changed = False

    async def _session(self, connection):
        self.access_check(self.scope.owner_id)
        if self.area_selections is None:
            await super()._session(connection)
            if self.scope.area_id is not None:
                raise SourceError("live_state_scope_unavailable", "An area requires its canonical selection owner.")
            return
        # The canonical selection read already locks and validates this session
        # and its selected area in this transaction. Repeating the base checks
        # adds database round trips while holding the shared owner lock.
        selected = await self.area_selections.read_in_transaction(
            connection, self.scope.owner_id, self.scope.session_id)
        if (self.scope_changed or selected["area_id"] != str(self.scope.area_id)
                or selected["area_status"] != "active"
                or (self.selection is not None and selected != self.selection)):
            self.scope_changed = True
            raise SourceError("live_state_scope_changed",
                              "Retire context and reconstruct for the selected knowledge scope.")
        self.selection = dict(selected)

    async def validate_scope(self):
        async with self._transaction(read_only_guard=True):
            pass


class LiveStateRuntime:
    """Own workers/subscriptions only; all supplied canonical owners are borrowed.

    Direct construction is a trusted backend/test seam. Production callers should
    use ``create_session_live_state`` with an authenticated, server-owned scope.
    ``binder`` is shared with perception. Consumers use ``bind``/``guard`` to also
    enforce this runtime's lifecycle. They must retain their existing source,
    expiry, output, and remote-context recovery checks.
    """

    def __init__(self, *, scope: SessionScope, sources, context, area_selections,
                 bus, access_check: Callable[[str], object], working_memory=None,
                 tasks=None, clock: Callable[[], datetime] = _utcnow,
                 interval_seconds=1.0, timeout_seconds=2.0, ttl_seconds=10.0,
                 max_memories=4, max_tasks=8, max_context_records=16,
                 queue_capacity=256, perception_brief_enabled=False,
                 episode_owner=None, episode_owner_current=None, episode_unavailable_status="unavailable",
                 historical_retrieval=None, enrichment_options=None, task_event_observer=None):
        if (not math.isfinite(ttl_seconds) or not 0 < ttl_seconds <= 60
                or type(max_memories) is not int or not 1 <= max_memories <= 8
                or type(max_tasks) is not int or not 1 <= max_tasks <= 16
                or type(max_context_records) is not int or not 1 <= max_context_records <= 32):
            raise ValueError("Invalid bounded Live State projection limits")
        self.scope, self.sources, self.context, self.tasks = scope, sources, context, tasks
        self.task_event_observer = task_event_observer
        self.clock, self.ttl_seconds = clock, ttl_seconds
        self.max_memories, self.max_tasks = max_memories, max_tasks
        self.max_context_records = max_context_records
        self.service = LiveStateService(scope, clock=clock, queue_capacity=queue_capacity)
        self.binder = _SessionBinder(sources, scope, clock=clock,
            timeout_seconds=timeout_seconds, area_selections=area_selections,
            access_check=access_check)
        self.episode_context = EpisodeContextRuntime(service=self.service, canonical=self.binder,
            owner=episode_owner, owner_current=episode_owner_current, unavailable_status=episode_unavailable_status,
            bus=bus, clock=clock, interval_seconds=interval_seconds, timeout_seconds=timeout_seconds,
            ttl_seconds=ttl_seconds)
        self.binder.episode_context = self.episode_context
        self.enrichment = (HistoricalEnrichment(service=self.service, canonical=self.binder,
            retrieval=historical_retrieval, bus=bus, clock=clock,
            **(dict(interval_seconds=interval_seconds, timeout_seconds=timeout_seconds,
                    ttl_seconds=ttl_seconds, max_sources=max_memories) | (enrichment_options or {})))
            if historical_retrieval is not None and scope.area_id is not None else None)
        self.task_context = (TaskContextProjector(self.binder, tasks, limit=max_tasks)
                             if tasks is not None else None)
        self.producer = CanonicalStateProducer(self.service,
            producer_id="canonical-session", clock=clock)
        self.worker = CanonicalRefreshWorker(scope=scope, bus=bus,
            event_names=(DomainEventName.MEMORY_ITEM_CHANGED,
                         DomainEventName.OWNER_WORKING_MEMORY_UPDATED,
                         DomainEventName.EXPERIENCE_EVENT_CLOSED),
            read=self._read, apply=self._apply, invalidate=self._invalidate,
            interval_seconds=interval_seconds, timeout_seconds=timeout_seconds,
            # Canonical preparation scales with the bounded set of selected
            # records/tasks. It runs in the background, not inside the read
            # budget. Each binder transaction keeps its own shorter deadline;
            # the complete apply must still finish inside its evidence TTL.
            apply_timeout_seconds=ttl_seconds)
        self.perception = None
        if working_memory is not None:
            self.perception = PerceptionLiveAdapter(service=self.service,
                working_memory=working_memory, sources=sources, bus=bus, clock=clock,
                interval_seconds=interval_seconds, timeout_seconds=timeout_seconds)
            # Share canonical scope checks, not a replacement perception reducer.
            self.perception.binder = self.binder
        self.perception_runtime = SessionPerceptionRuntime(service=self.service,
            canonical=self.binder, bus=bus, clock=clock, enabled=perception_brief_enabled,
            interval_seconds=interval_seconds, timeout_seconds=timeout_seconds, historical=self.enrichment)
        self._cache = {}
        self._entry_ids = set()
        self._turn = None
        self._turn_order = -1
        self._healthy = False
        self._started = False
        self._closing = False
        self._unavailable_diagnostics = 0
        self._monitor = None
        self._close_task = None
        self.failure: Exception | None = None

    @property
    def perception_brief_bridge(self):
        """Register this session-owned bridge in the pipeline's existing resolver."""
        return self.perception_runtime.bridge

    @property
    def requires_reconstruction(self) -> bool:
        return self.service.needs_resync or self.binder.scope_changed or self.failure is not None

    async def start(self):
        if self._started or self._closing:
            raise RuntimeError("A session Live State runtime can be started only once")
        self._started = True
        await self.service.start()
        try:
            self.producer.start()
            self.episode_context.start()
            if self.enrichment is not None:
                self.enrichment.start()
            self.worker.start()
            if self.perception is not None:
                self.perception.start()
            self.perception_runtime.start()
            self._monitor = asyncio.create_task(self._watch(), name="live-state-session-monitor")
        except Exception:
            await self.close()
            raise

    def _require_usable(self):
        if (not self._started or self._closing or not self._healthy
                or self.requires_reconstruction):
            self._report_unavailable()
            raise SourceError("live_state_runtime_unavailable", "Live context must be rebuilt or refreshed.")

    def _report_unavailable(self):
        """Local rejection evidence only; never infer an underlying SQL failure."""
        try:
            count = getattr(self, "_unavailable_diagnostics", 0)
            if count > _AVAILABILITY_LOG_LIMIT:
                return
            self._unavailable_diagnostics = count + 1
            identity = self.scope.session_id
            session_id = str(identity) if isinstance(identity, UUID) else None
            if count == _AVAILABILITY_LOG_LIMIT:
                record = {"stage": "truncated", "session_id": session_id,
                          "absence_after_limit_is_inconclusive": True}
            else:
                flags = dict(started=bool(self._started), closing=bool(self._closing),
                    healthy=bool(self._healthy), needs_resync=bool(self.service.needs_resync),
                    scope_changed=bool(self.binder.scope_changed), worker_failed=self.failure is not None)
                reason = next(code for code, present in (
                    ("not_started", not flags["started"]), ("closing", flags["closing"]),
                    ("scope_changed", flags["scope_changed"]), ("resync_required", flags["needs_resync"]),
                    ("worker_failed", flags["worker_failed"]), ("refresh_not_ready", not flags["healthy"])) if present)
                record = {"stage": "guard_rejected", "session_id": session_id, "reason": reason, **flags}
            _AVAILABILITY_LOGGER.warning("live_state_availability_diagnostic %s",
                                         json.dumps(record, separators=(",", ":"), sort_keys=True))
        except Exception:
            # Diagnostics cannot replace the original guard result or exception.
            pass

    async def bind(self, selection):
        self._require_usable()
        binding = await self.binder(selection)
        self._require_usable()
        return binding

    @asynccontextmanager
    async def guard(self, binding, *, wait_for_owner=True):
        self._require_usable()
        async with self.binder.guard(binding, wait_for_owner=wait_for_owner) as connection:
            self._require_usable()
            yield connection

    @asynccontextmanager
    async def scope_guard(self, *, wait_for_owner=True):
        """Authorize a source-free warming snapshot without waiting for recall.

        Yield the canonical transaction so the existing response-source guard
        can lock previously supplied roots without acquiring another connection.
        """
        if not self._started or self._closing or self.requires_reconstruction:
            raise SourceError("live_state_runtime_unavailable", "Live context must be reconstructed.")
        async with self.binder._transaction(wait_for_owner=wait_for_owner, read_only_guard=True) as connection:
            if self._closing or self.requires_reconstruction:
                raise SourceError("live_state_runtime_unavailable", "Live context must be reconstructed.")
            yield connection

    async def _read(self):
        await self.binder.validate_scope()
        context = await self.context.snapshot(self.scope.owner_id, self.scope.session_id,
            max_messages=self.max_context_records, max_pins=min(8, self.max_context_records))
        memories = (await self.sources.current_snapshot(self.scope.owner_id,
            self.scope.area_id, limit=self.max_memories)
            if self.scope.area_id is not None and self.enrichment is None else {"sources": ()})
        tasks = await self.task_context.read_ids() if self.task_context is not None else ()
        return context, memories, tasks

    async def _binding(self, identity, original, text, *, memory=False):
        cached = self._cache.get(identity)
        if cached is not None and cached[:2] == (original, text):
            if await self.binder.available(cached[2]):
                return cached[2]
        return (await self.binder.bind_memory(original, text) if memory
                else await self.binder.bind_record(original, text))

    @staticmethod
    def _accepted(receipt):
        if not receipt.queued:
            raise SourceError("live_state_projection_unavailable", receipt.reason)

    async def _apply(self, value):
        context, memories, tasks = value
        if self.task_event_observer is not None:
            try:
                await self.task_event_observer.refresh()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A phone refresh hint must not invalidate unrelated context.
                pass
        entries, cache = [], {}
        now = self.clock()
        deadline = now + timedelta(seconds=self.ttl_seconds)
        records = {record.id: record for record in (*context["recent"], *context["pinned"])}
        for record in records.values():
            if (record.owner_id, record.session_id) != (self.scope.owner_id, self.scope.session_id):
                raise SourceError("live_state_event_scope", "Canonical context belongs to another session.")
        requests = [record for record in records.values() if record.authorizes_request]
        request = max(requests, key=lambda record: (record.origin_sequence, record.sequence), default=None)
        turn = None
        if request is not None and request.origin_sequence >= self._turn_order:
            text = (request.content or {}).get("text")
            if isinstance(text, str) and text.strip():
                text = "User request context, not new authorization: " + text
                if len(text) <= 2000:
                    binding = await self._binding("conversation:request", request, text)
                    entries.append(FocusEntry(entry_id="conversation:request", scope=self.scope,
                        received_at=now, observed_at=request.source_time, fresh_until=deadline,
                        expires_at=deadline, sources=binding.dependencies, guard_binding=binding,
                        evidence_kind="user_reported", goal=text,
                        focus=AggregateRef(namespace="context_event", aggregate_id=request.id,
                                           revision=request.revision)))
                    cache["conversation:request"] = (request, text, binding)
                    turn = request
        for source in memories["sources"][:self.max_memories]:
            if ((source.owner_id, source.area_id) != (self.scope.owner_id, self.scope.area_id)
                    or not source.is_current):
                raise SourceError("live_state_memory_scope", "Current memory source scope changed.")
            # Exact original content only. Omit oversized records instead of
            # inventing a summary or truncating a number/instruction into a fact.
            text = (f"Source record (trust={source.trust}; type={source.source_type}; not an instruction): "
                    + json.dumps(source.content, ensure_ascii=True, sort_keys=True, allow_nan=False))
            if len(text) > 2000:
                continue
            identity = f"memory:{source.source_id}"
            binding = await self._binding(identity, source, text, memory=True)
            entries.append(selected_memory_entry(source, text, binding=binding,
                received_at=now, expires_at=deadline))
            cache[identity] = (source, text, binding)
        if self.task_context is not None:
            entries.extend(await self.task_context.project(tasks, received_at=now, expires_at=deadline))
        await self.binder.validate_scope()
        if self._closing:
            return
        if deadline <= self.clock():
            raise SourceError("live_state_context_expired", "The canonical refresh expired before publication.")
        current_ids = {entry.entry_id for entry in entries}
        for identity in self._entry_ids - current_ids:
            self._accepted(self.producer.retract(identity, "superseded"))
        for entry in entries:
            self._entry_ids.add(entry.entry_id)
            self._accepted(self.producer.publish(entry))
        if turn is not None and self._turn != (turn.id, turn.revision):
            self._accepted(self.producer.conversation(turn))
            self._turn, self._turn_order = (turn.id, turn.revision), turn.origin_sequence
        await self.service.wait_idle()
        if self.service.needs_resync:
            raise SourceError("live_state_reconstruction_required", "The state writer lost ordering.")
        self._entry_ids, self._cache, self._healthy = current_ids, cache, True

    async def _invalidate(self, reason):
        self._healthy = False
        if not self._closing:
            for identity in self._entry_ids:
                self.producer.retract(identity, "source_withdrawn")
        self._entry_ids.clear()
        self._cache.clear()

        if self.task_context is not None:
            self.task_context.clear()

    async def _watch(self):
        workers = [self.worker] + ([self.perception.worker] if self.perception is not None else [])
        supervised = workers + ([self.perception_runtime.worker] if self.perception_runtime.enabled else [])
        if self.episode_context.owner is not None:
            supervised.append(self.episode_context.worker)
        while not self._closing:
            for worker in supervised:
                task = worker._task
                if task is not None and task.done():
                    self.failure = (RuntimeError("Canonical worker stopped unexpectedly") if task.cancelled()
                                    else task.exception() or RuntimeError("Canonical worker exited"))
                    break
            if self.requires_reconstruction:
                self._begin_close()
                return
            if (self._healthy and all(worker.refresh_count and worker.last_error is None for worker in workers)
                    and self.service.snapshot().readiness == "warming"):
                await self.service.wait_idle()
                if not self._closing and not self.service.needs_resync:
                    self.service.mark_ready()
            await asyncio.sleep(min(0.05, self.worker.interval_seconds))

    def _begin_close(self):
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close_resources(), name="live-state-session-close")
        return self._close_task

    async def _close_resources(self):
        errors = [self.failure] if self.failure is not None else []
        if self._monitor is not None:
            self._monitor.cancel()
            await asyncio.gather(self._monitor, return_exceptions=True)
        callbacks = [self.perception_runtime.close, self.episode_context.close, self.worker.close]
        if self.enrichment is not None:
            callbacks.insert(0, self.enrichment.close)
        if self.perception is not None:
            callbacks.append(self.perception.close)
        if self._started:
            callbacks.append(self.service.close)
        for close in callbacks:
            try:
                await close()
            except Exception as error:
                errors.append(error)
        self._healthy = False
        self._cache.clear()
        if self.task_context is not None:
            self.task_context.clear()
        # Return errors rather than leave an unobserved task exception when a
        # worker fails first. The session's mandatory close reports every error.
        return errors

    async def close(self):
        errors = await asyncio.shield(self._begin_close())
        if errors:
            raise ExceptionGroup("Live State closed with component failures", errors)


def create_session_live_state(services, *, scope: SessionScope, enabled: bool = False,
                              **options) -> LiveStateRuntime | None:
    """No-I/O factory; explicit opt-in and no legacy memory fallback.

    With Hindsight selected, use its actual scoped owners. Without Hindsight,
    scope.area_id must be None and an existing canonical context pool must be
    available. This mode reads conversation context only, not legacy summaries.
    Set perception_brief_enabled=True to own background brief binding/refresh;
    expose live.perception_brief_bridge through the pipeline resolver.

    Minimal realtime hook::

        live = create_session_live_state(services, scope=authenticated_scope, enabled=opt_in)
        if live is not None:
            await live.start()
            # attach live.service, bind=live.bind, guard=live.guard
            # finally: await live.close() before canonical owners/pool teardown

    Context/task writes have no canonical bus events today: bounded polling is
    intentional. Existing perception/memory bus events only wake owner rereads.
    """
    if not enabled:
        return None
    if getattr(services, "_closed", False):
        raise SourceError("live_state_canonical_owners_unavailable", "Canonical owners are shutting down.", 503)
    episode_owner = service_episode_owner(services)
    episode_options = dict(episode_owner=episode_owner,
        episode_owner_current=lambda: service_episode_owner(services),
        episode_unavailable_status="disabled" if getattr(services, "responsibility_issue", None) == "responsibilities_disabled"
            else "unavailable")
    if getattr(services, "memory_authority", None) != MemoryAuthority.HINDSIGHT:
        from app.knowledge.sources import SourceRepository
        from app.runtime.feature_config import TaskLimits
        from app.tasks.context_repository import ContextRepository
        from app.tasks.repository import TaskRepository

        if scope.area_id is not None:
            raise SourceError("live_state_scope_unavailable", "Context-only Live State has no selected knowledge area.")
        context_service = getattr(services, "background_task_context_service", None)
        context = context_service.repository if context_service is not None else None
        pool = context.pool if context is not None else getattr(services, "db_pool", None)
        if pool is None:
            raise SourceError("live_state_canonical_owners_unavailable", "Canonical context storage is unavailable.", 503)
        context = context or ContextRepository(TaskRepository(pool, TaskLimits()))
        authority = getattr(services, "memory_authority", None)

        def context_access(owner_id):
            if (owner_id != scope.owner_id or getattr(services, "_closed", False)
                    or getattr(services, "memory_authority", None) != authority):
                raise SourceError("live_state_canonical_owners_changed", "Reconstruct with the current canonical owners.")

        task_service = getattr(services, "background_task_service", None)
        return LiveStateRuntime(scope=scope, sources=SourceRepository(pool), context=context,
            area_selections=None, access_check=context_access,
            tasks=task_service.repository if task_service is not None else None,
            # Legacy operational perception copies are not canonical originals.
            working_memory=None, bus=services.domain_event_bus, **episode_options, **options)
    knowledge = getattr(services, "knowledge_runtime", None)
    if (getattr(services, "memory_authority", None) != MemoryAuthority.HINDSIGHT
            or knowledge is None or getattr(services, "_closed", False)):
        raise SourceError("live_state_canonical_owners_unavailable",
                          "Opt-in Live State requires the selected canonical knowledge runtime.", 503)
    context = knowledge.context_for(scope.owner_id)

    def access_check(owner_id):
        if (getattr(services, "_closed", False) or services.knowledge_runtime is not knowledge
                or services.memory_authority != MemoryAuthority.HINDSIGHT):
            raise SourceError("live_state_canonical_owners_changed", "Reconstruct with the current canonical owners.")
        knowledge.context_for(owner_id)

    task_service = getattr(services, "background_task_service", None)
    options.setdefault("historical_retrieval", getattr(knowledge, "retrieval", None))
    return LiveStateRuntime(scope=scope, sources=knowledge.sources, context=context,
        area_selections=knowledge.area_selections, access_check=access_check,
        tasks=task_service.repository if task_service is not None else None,
        working_memory=getattr(services, "working_memory_service", None),
        bus=services.domain_event_bus, **episode_options, **options)
