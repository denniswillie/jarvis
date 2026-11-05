"""Canonical brief binding and session-owned refresh, without a model or task owner.

Existing context-event and selected knowledge-source provenance is bindable.
Explicit focus observations use their canonical check/episode versions, the
independent subject fence and an exact removable tool receipt. Binding an
assignment does not settle a check; the existing owners retain that authority.
"""
from __future__ import annotations

from datetime import datetime
from collections.abc import Callable
from contextlib import asynccontextmanager

import asyncpg

from app.knowledge.models import SourceError
from app.live_state.canonical import CONTEXT_EVENT, KNOWLEDGE_SOURCE, CanonicalContextBinder
from app.live_state.episode_context import EPISODE_NAMESPACES, link_episode_sources
from app.live_state.historical_enrichment import PREFIX as HISTORICAL_PREFIX
from app.live_state.models import GuardedPayloadBinding, LiveSnapshot
from app.live_state.perception import EvidenceReference, PerceptionBrief, PriorEvidence
from app.live_state.perception_focus import (
    CanonicalFocusInputs, FOCUS_CHECK, FOCUS_SUBJECT, focus_dependencies,
)
from app.live_state.refresh import CanonicalRefreshWorker
from app.perception.live_brief import LivePerceptionBriefBridge, perception_brief_document
from app.runtime.events import DomainEventName
from app.tasks.models import TaskError, json_object


SUPPORTED_NAMESPACES = (CONTEXT_EVENT, KNOWLEDGE_SOURCE)
OWNER_BINDING_GAPS = (
    "visual_focus_binding_unavailable",
    "responsibility_binding_unavailable",
    "situational_check_binding_unavailable",
    "episode_binding_unavailable",
)


class CanonicalPerceptionBinder:
    """Bind the full brief through existing removable payload/source contracts.

    Unlike a voice-text binding this stores the exact structured document, not
    a truncated string. The supplied canonical binder owns session/area checks,
    transaction ordering, source validation and provenance. No provider is used.
    """

    def __init__(self, canonical: CanonicalContextBinder, *, focus=None):
        self.canonical = canonical
        self.focus = focus
        self._cached = None

    def _focus_partition(self, brief):
        """Only question-bound focus episodes bypass the responsibility owner.

        The helper partitions, but never validates, references. Even a malformed
        focus question without its subject dependency must reach the focus
        validator rather than quietly becoming an ordinary context-only brief.
        """
        selected = focus_dependencies(brief)
        focus_namespaces = (FOCUS_CHECK, FOCUS_SUBJECT)
        has_focus = bool(selected) or any(
            ref.source_namespace in focus_namespaces for ref in brief.dependencies
        ) or any(
            ref is not None and ref.source_namespace in focus_namespaces
            for question in brief.questions for ref in (question.check_ref, question.focus_ref)
        )
        if has_focus and self.focus is None:
            raise SourceError("perception_brief_dependency_unsupported",
                              "The focus references need their canonical owner adapter.")
        if any(ref.source_namespace in focus_namespaces and ref not in selected
               for ref in brief.dependencies) or len(selected) != len(set(selected)):
            raise SourceError("perception_focus_dependency_invalid",
                              "Focus references must belong to the original assigned question.")
        return selected, tuple(ref for ref in brief.dependencies if ref not in selected), has_focus

    def _check(self, brief):
        if brief.scope != self.canonical.scope:
            raise SourceError("perception_brief_scope_mismatch", "The brief belongs to another session scope.")
        if not brief.created_at <= self.canonical.clock() < brief.expires_at:
            raise SourceError("perception_brief_expired", "The assignment is not current.")
        _, remaining, _ = self._focus_partition(brief)
        supported = SUPPORTED_NAMESPACES
        if self.canonical.episode_context is not None and self.canonical.episode_context.owner is not None:
            supported += tuple(EPISODE_NAMESPACES)
        if any(source.source_namespace not in supported for source in remaining):
            raise SourceError("perception_brief_dependency_unsupported",
                              "The referenced owner has no supported canonical brief binding.")

    async def _validate_dependencies(self, connection, brief):
        _, remaining, has_focus = self._focus_partition(brief)
        episodes = await self.canonical._validate_dependencies(connection, remaining)
        if episodes:
            # Owner revisions alone do not prove a conditional cue is still
            # relevant. Recheck its original evidence and current activity in
            # this same transaction, both when binding and before effects.
            await self.canonical.episode_context.validate_assignment(connection, brief, episodes)
        if has_focus:
            await self.focus.validate(connection, brief)
        return episodes

    @asynccontextmanager
    async def guard(self, brief: PerceptionBrief, binding: GuardedPayloadBinding, *, effect_payload_ids=()):
        """Current assignment applicability; not a check-resolution operation.

        Historical retention may keep the old receipt, but owner/effect consumers
        must use this guard (or their equivalent canonical fence) before applying
        a finding. A payload-only guard cannot validate focus replacement.
        """
        self._check(brief)
        if (binding.scope != brief.scope or binding.dependencies != brief.dependencies
                or len(binding.payload_root_ids) != 1):
            raise SourceError("perception_brief_binding_mismatch", "The binding is not for this assignment.")
        async with self.canonical._transaction() as connection:
            if any(ref.source_namespace in EPISODE_NAMESPACES for ref in brief.dependencies):
                await self._validate_dependencies(connection, brief)
            await self.canonical._lock_roots(connection, (*binding.payload_root_ids, *effect_payload_ids))
            raw = await connection.fetchval(
                "select content from jarvis_context_payloads where owner_id=$1 and id=$2",
                brief.scope.owner_id, binding.payload_root_ids[0])
            if json_object(raw) != {"format": "jarvis.perception_brief.v1", "brief": perception_brief_document(brief)}:
                raise SourceError("perception_brief_binding_mismatch", "The stored assignment differs.")
            await self._validate_dependencies(connection, brief)
            self._check(brief)
            yield connection

    async def bind(self, brief: PerceptionBrief) -> GuardedPayloadBinding:
        self._check(brief)
        document = perception_brief_document(brief)
        canonical = self.canonical
        async with canonical._transaction(derivation=True) as connection:
            episodes = await self._validate_dependencies(connection, brief)
            if self._cached is not None and self._cached[0] == document:
                binding = self._cached[1]
                await canonical._lock_roots(connection, binding.payload_root_ids)
                self._check(brief)
                return binding
            event_dependencies = tuple(source for source in brief.dependencies
                                       if source.source_namespace == CONTEXT_EVENT)
            records = (await canonical.context._records(connection, brief.scope.owner_id,
                       [source.source_id for source in event_dependencies])) if event_dependencies else {}
            for record in records.values():
                if ((record.owner_id, record.session_id) != (brief.scope.owner_id, brief.scope.session_id)
                        or record.availability != "available" or record.content is None):
                    raise SourceError("perception_brief_context_unavailable", "Canonical context is unavailable in this session.")
            episode_roots = {root for episode in episodes for root in episode.source_payload_ids}
            event_roots = tuple(sorted({record.payload_id for record in records.values()} | episode_roots))
            if event_roots:
                await canonical._lock_roots(connection, event_roots)
            # Context availability/revisions are checked again after locking.
            await self._validate_dependencies(connection, brief)
            memory_dependencies = tuple(source for source in brief.dependencies
                                        if source.source_namespace == KNOWLEDGE_SOURCE)
            memories = (await canonical.sources._read_many(connection, brief.scope.owner_id,
                brief.scope.area_id, [(source.source_id, source.revision) for source in memory_dependencies])
                if memory_dependencies else ())

            async def materialize(conn, originals):
                self._check(brief)
                root = await canonical.tasks.payload(conn, brief.scope.owner_id,
                    {"format": "jarvis.perception_brief.v1", "brief": document},
                    source_payload_ids=tuple(sorted(set(event_roots) | set(originals))))
                if event_dependencies:
                    await conn.executemany(
                        """insert into jarvis_context_event_dependencies(owner_id,dependent_payload_id,
                           event_id,event_revision,source_payload_id,source_usage) values($1,$2,$3,$4,$5,$6)""",
                        [(brief.scope.owner_id, root, source.source_id, source.revision,
                          records[source.source_id].payload_id, records[source.source_id].dependency_usage)
                         for source in event_dependencies])
                await link_episode_sources(canonical, conn, root, episodes)
                receipt = GuardedPayloadBinding(scope=brief.scope, payload_root_ids=(root,),
                                                dependencies=brief.dependencies)
                return receipt, root

            if memories:
                binding = await canonical.provenance.bind_in_transaction(connection,
                    brief.scope.owner_id, brief.scope.area_id,
                    [source.original_evidence() for source in memories], materialize,
                    **({"shared_task_owner": True} if canonical.shared_derivations else {}))
            else:
                binding, _ = await materialize(connection, ())
        self._cached = (document, binding)
        return binding


class SessionPerceptionRuntime:
    """Own a background worker and expose one borrowed-by-pipeline bridge.

    Minimal briefs need only the canonical session and payload graph, not
    Hindsight. Current guarded observation entries may contribute prior evidence.
    Canonical explicit-focus receipts can supply observational questions, never
    obligations or resolved checks. Snapshot focus text is not authority. Analyzer
    completion events do not trigger refresh.
    """

    def __init__(self, *, service, canonical: CanonicalContextBinder, bus,
                 clock: Callable[[], datetime], enabled=False,
                 interval_seconds=1.0, timeout_seconds=2.0, historical=None):
        self.service, self.clock, self.enabled = service, clock, enabled
        self.historical = historical
        self.canonical = canonical
        self.scope_epoch = service.snapshot().cursor.epoch
        self.focus_inputs = CanonicalFocusInputs(canonical)
        self.binder = CanonicalPerceptionBinder(canonical, focus=self.focus_inputs)
        self._focus = None
        self.focus_status = "warming"
        self.focus_omission = None
        self.bridge = LivePerceptionBriefBridge(scope=canonical.scope,
            scope_epoch=self.scope_epoch, enabled=enabled, clock=clock, result_guard=self.binder.guard,
            allow_warming_minimal=True)
        self.worker = CanonicalRefreshWorker(scope=canonical.scope, bus=bus,
            event_names=(DomainEventName.MEMORY_ITEM_CHANGED, DomainEventName.OWNER_WORKING_MEMORY_UPDATED),
            read=self._read, apply=self._apply, invalidate=self._invalidate,
            interval_seconds=interval_seconds, timeout_seconds=timeout_seconds)
        self._started = False
        self._closed = False
        self._ready = False
        self.omissions: tuple[str, ...] = ()

    def start(self):
        if self._started or self._closed:
            raise RuntimeError("A perception runtime can be started only once")
        self._started = True
        if self.enabled:
            self.worker.start()

    async def _read(self):
        snapshot = self.service.snapshot()
        if snapshot.cursor.epoch != self.scope_epoch:
            raise SourceError("perception_brief_epoch_changed", "Reconstruct after Live State recovery.")
        priors, omissions = [], []
        now = self.clock()
        focus = None
        self.focus_status, self.focus_omission = "warming", None
        if snapshot.readiness == "ready":
            try:
                focus = await self.focus_inputs.read()
                self.focus_status = "available" if focus is not None else "no_active_focus"
            except (SourceError, TaskError, TimeoutError, asyncpg.PostgresError, asyncpg.InterfaceError) as error:
                self.focus_status = "unavailable"
                self.focus_omission = getattr(error, "code", "perception_focus_storage_unavailable")
                omissions.append(self.focus_omission)
            for entry in snapshot.entries:
                if entry.kind != "observation":
                    continue
                if (entry.validity != "active" or entry.guard_binding is None or len(entry.sources) != 1
                        or entry.sources[0].source_namespace not in SUPPORTED_NAMESPACES):
                    omissions.append("observation_provenance_unsupported")
                    continue
                if (entry.observed_at is None or entry.observed_at > now or entry.confidence is None
                        or any(value is not None and value <= now for value in (entry.fresh_until, entry.expires_at))):
                    omissions.append("observation_time_or_confidence_unavailable")
                    continue
                uncertainty = "\n".join(entry.unknowns) or "Prior analyzer interpretation, not proof of a completed action."
                if len(entry.summary) > 1000 or len(uncertainty) > 1000 or len(priors) >= 4:
                    omissions.append("prior_evidence_budget")
                    continue
                if not await self.canonical.available(entry.guard_binding):
                    omissions.append("observation_original_unavailable")
                    continue
                priors.append(PriorEvidence(reference=EvidenceReference(source=entry.sources[0],
                    captured_at=entry.observed_at), finding=entry.summary,
                    confidence=entry.confidence, uncertainty=uncertainty))
        historical = self.historical.current_context(snapshot) if self.historical is not None else None
        historical_dependencies = []
        if historical is not None:
            for entry in snapshot.entries:
                if entry.kind != "selected_memory" or not entry.entry_id.startswith(HISTORICAL_PREFIX):
                    continue
                memories = tuple(source for source in entry.sources if source.source_namespace == KNOWLEDGE_SOURCE)
                if (entry.validity != "active" or entry.guard_binding is None or len(memories) != 1
                        or entry.observed_at is None or entry.observed_at > now or len(entry.summary) > 1000
                        or len(priors) >= 8
                        or any(t is not None and t <= now for t in (entry.fresh_until, entry.expires_at))
                        or not set(historical.dependencies).issubset(entry.sources)
                        or not await self.canonical.available(entry.guard_binding)):
                    omissions.append("historical_evidence_unavailable_or_budget")
                    continue
                priors.append(PriorEvidence(reference=EvidenceReference(source=memories[0],
                    captured_at=entry.observed_at), finding=entry.summary, confidence=0,
                    uncertainty="Historical source context, not camera evidence. No current visual confidence is implied."))
                historical_dependencies.extend(entry.sources)
        self.omissions = tuple(omissions)
        episode_context = self.canonical.episode_context
        status = episode_context.status if episode_context is not None else "unavailable"
        capabilities = snapshot.capabilities.model_copy(update={"durable_responsibilities": status})
        value = snapshot.model_copy(update={"capabilities": capabilities}), tuple(priors), focus
        return (*value, (historical, tuple(dict.fromkeys(historical_dependencies)))) if self.historical is not None else value

    async def _apply(self, value):
        snapshot, priors, focus, *extra = value
        historical, history_dependencies = extra[0] if extra else (None, ())
        if self.historical is not None and historical is not self.historical.current_context():
            priors = tuple(prior for prior in priors if prior.reference.source.source_namespace != KNOWLEDGE_SOURCE)
            historical, history_dependencies = None, ()
        self._ready = False
        episode_context = self.canonical.episode_context
        checks = episode_context.checks(snapshot) if episode_context is not None else ()
        if focus is not None:
            # Typed explicit questions require their canonical check projection.
            # This is the existing focus owner's lifecycle, not a new admission
            # or a claim that responsibility ownership is available.
            focus_check = focus.focus.as_check(self.canonical.scope, now=self.clock())
            if focus_check.relevance_now:
                checks = (*checks, focus_check)
            else:
                # A resolved/expired focus must not leave orphan dependencies
                # when the composer correctly excludes its inactive question.
                focus = None
        self._focus = focus
        dependencies = {prior.reference.source for prior in priors}
        for check in checks:
            dependencies.update(check.dependency_versions)
        if focus is not None:
            dependencies.update(focus.dependencies)
        context_dependencies = tuple(dict.fromkeys((*(focus.dependencies if focus else ()),
            *history_dependencies, *(historical.dependencies if historical is not None else ()))))
        dependencies.update(context_dependencies)

        def eligible(source):
            if source.source_namespace == FOCUS_SUBJECT:
                return (self._focus == focus and focus is not None and self.clock() < focus.expires_at
                        and source == focus.question.focus_ref)
            return source in dependencies

        try:
            composition = await self.bridge.refresh(snapshot, bind=self.binder.bind,
                checks=checks,
                prior_evidence=priors, explicit_questions=(focus.question,) if focus else (),
                context_dependencies=context_dependencies, dependency_eligible=eligible,
                intent_ref=historical.intent_ref if historical else None,
                activity=historical.activity if historical else None)
        except SourceError as error:
            if not error.code.startswith(("perception_focus_", "live_episode_")):
                raise
            # Replacement between read and bind must not remove valid priors or
            # block the minimal scene-transition assignment. Never retry the old
            # question; the next bounded owner read can select its replacement.
            self._focus = None
            self.focus_status, self.focus_omission = "unavailable", error.code
            self.omissions = (*self.omissions, error.code)
            prior_dependencies = {prior.reference.source for prior in priors}
            composition = await self.bridge.refresh(snapshot, bind=self.binder.bind,
                prior_evidence=priors, dependency_eligible=lambda source: source in prior_dependencies)
        self._ready = composition is not None and self.bridge.last_omission is None

    async def _invalidate(self, reason):
        self._ready = False
        self._focus = None
        # Public bridge refresh clears its binding without fabricating source
        # removal, resolving checks, or resetting the canonical state epoch.
        snapshot = self.service.snapshot().model_copy(update={"readiness": "degraded"})
        await self.bridge.refresh(snapshot, bind=self.binder.bind)

    def diagnostics(self) -> dict:
        episodes = self.canonical.episode_context
        status = episodes.status if episodes is not None else "unavailable"
        gaps = tuple(gap for gap in OWNER_BINDING_GAPS if status != "available" or gap == "visual_focus_binding_unavailable")
        return {
            "status": "closed" if self._closed else "disabled" if not self.enabled else
                      "ready" if self._ready else "unavailable" if self.worker.last_error else "warming",
            "bindable_namespaces": (CONTEXT_EVENT, KNOWLEDGE_SOURCE) if self.canonical.scope.area_id is not None else (CONTEXT_EVENT,),
            "selected_memory_binding": "supported" if self.canonical.scope.area_id is not None else "disabled",
            "historical_enrichment": self.historical.status if self.historical is not None else "disabled",
            "durable_responsibilities": status, "owner_binding_gaps": gaps,
            "episode_check_projection": "canonical_owner_guarded" if status == "available" else "unavailable",
            "episode_omissions": episodes.omissions if episodes is not None else (),
            "episode_visual_resolution": "unavailable",
            "visual_focus_input": self.focus_status, "visual_focus_omission": self.focus_omission,
            "visual_focus_subject_binding": "supported_with_canonical_receipt",
            "visual_focus_check_episode_binding": "supported_with_canonical_receipt",
            "omissions": self.omissions, "failure_count": self.worker.failure_count,
            "last_error": self.worker.last_error,
        }

    async def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            await self.worker.close()
        finally:
            self._ready = False
            self.bridge.close()
