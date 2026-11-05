"""Canonical binding for Live State copies, outside the conversation fast path.

No new source authority lives here. Existing payload/source dependencies retain
their removal, correction and historical-read semantics. A successful bind is
not permanent permission: use ``guard`` at the actual provider-write boundary.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
import math

from app.knowledge.models import CanonicalOwnerBusy, SourceError, SourceVersion
from app.guard_diagnostics import guard_wait_stage
from app.knowledge.read_provenance import SourceReadProvenance
from app.live_state.models import GuardedPayloadBinding, SessionScope, SourceDependency
from app.live_state.projection import VoiceSelection
from app.tasks.context_events import ContextRecord
from app.tasks.context_repository import ContextRepository


KNOWLEDGE_SOURCE = "knowledge_source"
CONTEXT_EVENT = "context_event"


def memory_dependency(source: SourceVersion) -> SourceDependency:
    return SourceDependency(source_namespace=KNOWLEDGE_SOURCE,
        source_id=source.source_id, revision=source.revision, generation=source.generation)


def context_dependency(record: ContextRecord) -> SourceDependency:
    return SourceDependency(source_namespace=CONTEXT_EVENT,
        source_id=record.id, revision=record.revision)


class CanonicalContextBinder:
    def __init__(self, sources, scope: SessionScope, *, clock: Callable[[], datetime],
                 timeout_seconds: float = 2.0):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Binding timeout must be finite and positive")
        self.sources, self.scope, self.clock = sources, scope, clock
        self.timeout_seconds = timeout_seconds
        self.provenance = SourceReadProvenance(sources)
        self.tasks = self.provenance.tasks
        self.context = ContextRepository(self.tasks)
        self.episode_context = None

    def _scope(self, scope):
        if scope != self.scope:
            raise SourceError("live_state_scope_changed", "Live context belongs to another scope.")

    async def _session(self, connection):
        if not await connection.fetchval(
                "select true from jarvis_sessions where owner_id=$1 and id=$2 for key share",
                self.scope.owner_id, self.scope.session_id):
            raise SourceError("live_state_session_unavailable", "The conversation session is unavailable.")
        if self.scope.area_id is not None:
            await self.sources._area(connection, self.scope.owner_id, self.scope.area_id)

    @property
    def shared_derivations(self):
        return callable(getattr(self.sources, "derivation_transaction", None))

    @asynccontextmanager
    async def _transaction(self, *, wait_for_owner=True, read_only_guard=False, derivation=False):
        if (any(type(value) is not bool for value in (wait_for_owner, read_only_guard, derivation))
                or (read_only_guard and derivation)):
            raise ValueError("Owner lock waiting and transaction mode must be explicit")
        async with asyncio.timeout(self.timeout_seconds):
            # Older borrowed adapters remain conservatively exclusive. The real
            # source repository exposes compatible shared memory/task ownership.
            shared_factory = getattr(self.sources,
                "derivation_transaction" if derivation else "read_guard_transaction", None)
            shared = (read_only_guard or derivation) and callable(shared_factory)
            factory = shared_factory if shared else self.sources.transaction
            transaction = (factory(self.scope.owner_id) if wait_for_owner else
                           factory(self.scope.owner_id, wait_for_owner=False))
            async with transaction as connection:
                with guard_wait_stage("task_owner_lock"):
                    if wait_for_owner:
                        lock = self.tasks.lock_owner_shared if shared else self.tasks.lock_owner
                        await lock(connection, self.scope.owner_id)
                    else:
                        lock = self.tasks.try_lock_owner_shared if shared else self.tasks.try_lock_owner
                        if not await lock(connection, self.scope.owner_id):
                            raise CanonicalOwnerBusy()
                with guard_wait_stage("scope_revalidation"):
                    await self._session(connection)
                yield connection

    async def _lock_roots(self, connection, roots):
        with guard_wait_stage("canonical_source_lock"):
            if not roots or not await connection.fetchval(
                    "select lock_jarvis_copy_sources($1,$2::uuid[])",
                    self.scope.owner_id, sorted(set(roots))):
                raise SourceError("live_state_context_revoked", "The context originals are unavailable.")

    async def _validate_dependencies(self, connection, dependencies):
        from app.live_state.episode_context import EPISODE_NAMESPACES, validate_episode_dependencies
        from app.live_state.task_context import TaskSnapshotControlChange, TaskSnapshotProgress, validate_task_dependency
        episodes = await validate_episode_dependencies(self, connection, dependencies)
        memories, events = [], []
        progress = None
        for dependency in dependencies:
            if dependency.source_namespace == KNOWLEDGE_SOURCE:
                if dependency.generation is None or self.scope.area_id is None:
                    raise SourceError("live_state_source_scope", "Memory evidence needs its area and generation.")
                memories.append(dependency)
            elif dependency.source_namespace == CONTEXT_EVENT:
                if dependency.generation is not None:
                    raise SourceError("live_state_source_version", "Context events have no lifecycle generation.")
                events.append(dependency)
            elif dependency.source_namespace == "task_state_snapshot":
                # A snapshot carries a real event sequence, not a fabricated
                # task revision. Its complete state/lineage is reread under the
                # same canonical transaction at every provider-write guard.
                try:
                    await validate_task_dependency(self, connection, dependency)
                except (TaskSnapshotProgress, TaskSnapshotControlChange) as error:
                    # Do not hide a later correction/permission/source failure
                    # behind an otherwise recoverable task-state supersession.
                    progress = error
            elif dependency.source_namespace in EPISODE_NAMESPACES:
                # Already checked as complete owner/check/episode relationships.
                pass
            else:
                raise SourceError("live_state_source_unsupported", "This source needs its canonical adapter.")
        if memories:
            # Explicit old revisions can be legitimate historical evidence.
            # Parent payloads retain their independently recorded source_usage;
            # their guards, not this lookup, authorize historical interpretation.
            resolved = await self.sources._read_many(connection, self.scope.owner_id,
                self.scope.area_id, [(item.source_id, item.revision) for item in memories])
            if any(memory_dependency(value) != expected
                   for value, expected in zip(resolved, memories, strict=True)):
                raise SourceError("live_state_source_changed", "Selected source generations changed.")
        if events:
            records = await self.context._records(connection, self.scope.owner_id,
                [item.source_id for item in events])
            for dependency in events:
                record = records[dependency.source_id]
                if (record.session_id != self.scope.session_id
                        or record.availability != "available"
                        or context_dependency(record) != dependency):
                    raise SourceError("live_state_event_changed", "Selected conversation evidence changed.")
        if progress is not None:
            raise progress
        return episodes

    async def _materialize(self, connection, text, dependencies, parents):
        if not isinstance(text, str) or not text or len(text) > 4096 or "\x00" in text:
            raise ValueError("A bounded nonempty context body is required")
        identity = await self.tasks.payload(connection, self.scope.owner_id, {
            "format": "jarvis.live_state.voice.v1", "scope": self.scope.model_dump(mode="json"),
            "text": text, "dependencies": [item.model_dump(mode="json") for item in dependencies],
        })
        if parents:
            await connection.executemany(
                """insert into jarvis_context_payload_dependencies
                   (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3)""",
                [(self.scope.owner_id, parent, identity) for parent in sorted(set(parents))])
        return GuardedPayloadBinding(scope=self.scope, payload_root_ids=(identity,),
            dependencies=tuple(dependencies))

    async def bind_memory(self, source: SourceVersion, text: str) -> GuardedPayloadBinding:
        if source.owner_id != self.scope.owner_id or source.area_id != self.scope.area_id:
            raise SourceError("live_state_memory_scope", "Selected memory belongs to another scope.")
        dependency = memory_dependency(source)
        async with self._transaction(derivation=True) as connection:
            async def materialize(conn, originals):
                binding = await self._materialize(conn, text, (dependency,), originals)
                return binding, binding.payload_root_ids[0]
            return await self.provenance.bind_in_transaction(connection, self.scope.owner_id,
                self.scope.area_id, [source.original_evidence()], materialize,
                **({"shared_task_owner": True} if self.shared_derivations else {}))

    async def bind_record(self, record: ContextRecord, text: str) -> GuardedPayloadBinding:
        if (record.owner_id, record.session_id) != (self.scope.owner_id, self.scope.session_id):
            raise SourceError("live_state_event_scope", "Conversation evidence belongs to another session.")
        async with self._transaction(derivation=True) as connection:
            await self._lock_roots(connection, (record.payload_id,))
            fresh = await self.context._record(connection, self.scope.owner_id, record.id)
            if fresh != record or fresh.availability != "available" or fresh.content is None:
                raise SourceError("live_state_event_changed", "Conversation evidence changed during binding.")
            binding = await self._materialize(connection, text, (context_dependency(fresh),),
                (fresh.payload_id,))
            await connection.execute(
                """insert into jarvis_context_event_dependencies(owner_id,dependent_payload_id,
                   event_id,event_revision,source_payload_id,source_usage) values($1,$2,$3,$4,$5,$6)""",
                self.scope.owner_id, binding.payload_root_ids[0], fresh.id, fresh.revision,
                fresh.payload_id, fresh.dependency_usage)
            return binding

    async def __call__(self, selection: VoiceSelection) -> GuardedPayloadBinding:
        self._scope(selection.scope)
        if selection.expires_at is not None and selection.expires_at <= self.clock():
            raise SourceError("live_state_context_expired", "The selected context expired before binding.")
        if len(selection.dependencies) > 64 or len(selection.parent_payload_root_ids) > 64:
            raise ValueError("Context binding exceeds its source budget")
        async with self._transaction(derivation=True) as connection:
            # Episode owner -> original payload ordering matches owner mutations.
            from app.live_state.episode_context import EPISODE_NAMESPACES
            if any(ref.source_namespace in EPISODE_NAMESPACES for ref in selection.dependencies):
                await self._validate_dependencies(connection, selection.dependencies)
            await self._lock_roots(connection, selection.parent_payload_root_ids)
            await self._validate_dependencies(connection, selection.dependencies)
            if selection.expires_at is not None and selection.expires_at <= self.clock():
                raise SourceError("live_state_context_expired", "The selected context expired during binding.")
            return await self._materialize(connection, selection.text, selection.dependencies,
                selection.parent_payload_root_ids)

    @asynccontextmanager
    async def guard(self, binding: GuardedPayloadBinding, *, wait_for_owner=True):
        self._scope(binding.scope)
        async with self._transaction(wait_for_owner=wait_for_owner, read_only_guard=True) as connection:
            from app.live_state.episode_context import EPISODE_NAMESPACES
            if any(ref.source_namespace in EPISODE_NAMESPACES for ref in binding.dependencies):
                with guard_wait_stage("dependency_revalidation"):
                    await self._validate_dependencies(connection, binding.dependencies)
            await self._lock_roots(connection, binding.payload_root_ids)
            with guard_wait_stage("dependency_revalidation"):
                await self._validate_dependencies(connection, binding.dependencies)
            yield connection

    async def available(self, binding: GuardedPayloadBinding) -> bool:
        try:
            async with self.guard(binding):
                return True
        except (SourceError, TimeoutError):
            return False
