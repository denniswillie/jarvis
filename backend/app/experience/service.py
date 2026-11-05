from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID
from weakref import WeakValueDictionary

from app.experience.evidence import (
    EventEvidenceDescriptor,
    event_kind_for_modalities,
    evidence_from_message,
    evidence_from_observation,
    evidence_from_tool_execution,
    reduce_event_aggregate,
)
from app.experience.keyframes import (
    DeterministicKeyframeSelector,
    KeyframeRetentionPolicy,
)
from app.experience.models import ExperienceEvent, ToolExecution
from app.experience.repository import (
    ExperienceConflictError,
    ExperienceEventClosedError,
    ExperienceRepository,
    ExperienceSessionMissingError,
)
from app.experience.segmenter import (
    ExperienceEventSegmenter,
    event_last_evidence_at,
)
from app.perception.models import VisualObservation
from app.perception.services import WorkingMemoryService
from app.runtime.events import (
    DomainEvent,
    DomainEventBus,
    DomainEventName,
    ExperienceEventClosedPayload,
)
from app.sessions.models import JarvisMessage

LOGGER = logging.getLogger(__name__)
SESSION_CLOSE_DRAIN_TIMEOUT_SECONDS = 10.0
MAX_BACKGROUND_TOOL_ATTACHMENTS = 128


@dataclass(frozen=True)
class EvidenceAttachmentOutcome:
    attached: bool
    event: ExperienceEvent | None
    reason: str


class ExperienceEventService:
    def __init__(
        self,
        *,
        repository: ExperienceRepository,
        segmenter: ExperienceEventSegmenter,
        domain_event_bus: DomainEventBus | None = None,
        working_memory_service: WorkingMemoryService | None = None,
        keyframe_selector: DeterministicKeyframeSelector | None = None,
        keyframe_policy: KeyframeRetentionPolicy | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.repository = repository
        self.segmenter = segmenter
        self._domain_event_bus = domain_event_bus
        self._working_memory_service = working_memory_service
        self._keyframe_selector = keyframe_selector
        self._keyframe_policy = keyframe_policy
        self._now = now
        self._session_locks: WeakValueDictionary[
            tuple[str, UUID], asyncio.Lock
        ] = WeakValueDictionary()
        self._activity_condition = asyncio.Condition()
        self._active_counts: defaultdict[tuple[str, UUID], int] = defaultdict(int)
        self._close_tasks: dict[tuple[str, UUID], asyncio.Task[None]] = {}
        self._tool_attachment_tasks: dict[tuple[str, UUID, UUID], asyncio.Task[None]] = {}
        self._closing = False
        self._shutdown_task: asyncio.Task[None] | None = None

    @asynccontextmanager
    async def activity(self, *, owner_id: str, session_id: UUID) -> AsyncIterator[None]:
        key = (owner_id, session_id)
        async with self._activity_condition:
            self._active_counts[key] += 1
        try:
            yield
        finally:
            async with self._activity_condition:
                self._active_counts[key] -= 1
                if self._active_counts[key] <= 0:
                    self._active_counts.pop(key, None)
                self._activity_condition.notify_all()

    async def attach_observation(
        self, observation: VisualObservation
    ) -> EvidenceAttachmentOutcome:
        async with self.activity(
            owner_id=observation.owner_id,
            session_id=observation.session_id,
        ):
            outcome = await self._attach(
                owner_id=observation.owner_id,
                session_id=observation.session_id,
                evidence=evidence_from_observation(observation),
            )
            await self._retain_observation_candidates(observation, outcome)
            await self._update_working_memory_pointer(
                owner_id=observation.owner_id,
                session_id=observation.session_id,
                outcome=outcome,
            )
            return outcome

    async def attach_message(
        self,
        *,
        owner_id: str,
        message: JarvisMessage,
    ) -> EvidenceAttachmentOutcome:
        async with self.activity(owner_id=owner_id, session_id=message.session_id):
            outcome = await self._attach(
                owner_id=owner_id,
                session_id=message.session_id,
                evidence=evidence_from_message(message),
            )
            await self._update_working_memory_pointer(
                owner_id=owner_id,
                session_id=message.session_id,
                outcome=outcome,
            )
            return outcome

    async def attach_tool_execution(
        self, execution: ToolExecution
    ) -> EvidenceAttachmentOutcome:
        async with self.activity(
            owner_id=execution.owner_id,
            session_id=execution.session_id,
        ):
            outcome = await self._attach(
                owner_id=execution.owner_id,
                session_id=execution.session_id,
                evidence=evidence_from_tool_execution(execution),
            )
            await self._update_working_memory_pointer(
                owner_id=execution.owner_id,
                session_id=execution.session_id,
                outcome=outcome,
            )
            return outcome

    async def defer_tool_execution(self, execution: ToolExecution) -> bool:
        """Enqueue optional grouping of an already durable terminal execution.

        Admission is not an attachment receipt. Reserve activity before returning
        so a concurrent session close cannot overtake the background task.
        """
        key = (execution.owner_id, execution.session_id, execution.id)
        session_key = key[:2]
        async with self._activity_condition:
            if self._closing or execution.status not in {"completed", "failed"}:
                LOGGER.warning("tool experience attachment not queued reason=closed_or_nonterminal")
                return False
            if key in self._tool_attachment_tasks:
                return True
            if len(self._tool_attachment_tasks) >= MAX_BACKGROUND_TOOL_ATTACHMENTS:
                LOGGER.warning("tool experience attachment not queued reason=queue_full")
                return False
            self._active_counts[session_key] += 1
            task = asyncio.create_task(self._attach_deferred_tool(execution, session_key))
            self._tool_attachment_tasks[key] = task
            task.add_done_callback(lambda completed, identity=key:
                self._finish_tool_attachment(identity, completed))
        return True

    async def _attach_deferred_tool(self, execution, session_key):
        try:
            async with asyncio.timeout(SESSION_CLOSE_DRAIN_TIMEOUT_SECONDS):
                await self.attach_tool_execution(execution)
        except asyncio.CancelledError:
            LOGGER.warning("background tool experience attachment interrupted")
            raise
        except Exception as error:
            # The durable execution remains the truth; never retry its handler.
            LOGGER.warning("background tool experience attachment failed error_type=%s",
                type(error).__name__)
        finally:
            async with self._activity_condition:
                self._active_counts[session_key] -= 1
                if self._active_counts[session_key] <= 0:
                    self._active_counts.pop(session_key, None)
                self._activity_condition.notify_all()

    def _finish_tool_attachment(self, key, task):
        if self._tool_attachment_tasks.get(key) is task:
            self._tool_attachment_tasks.pop(key, None)
        if not task.cancelled():
            task.exception()

    def schedule_session_close(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        closed_at: datetime,
    ) -> None:
        if self._closing:
            return
        key = (owner_id, session_id)
        existing = self._close_tasks.get(key)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(
            self._close_after_drain(
                owner_id=owner_id,
                session_id=session_id,
                closed_at=closed_at,
            )
        )
        self._close_tasks[key] = task
        task.add_done_callback(
            lambda completed, close_key=key: self._finish_close_task(
                close_key,
                completed,
            )
        )

    async def sweep_due_events(self, *, now: datetime | None = None) -> int:
        current_time = now or self._now()
        events = await self.repository.list_open_events_due(
            idle_cutoff=current_time
            - self.segmenter.policy_idle_delta,
            max_started_cutoff=current_time
            - self.segmenter.policy_max_delta,
            limit=100,
        )
        closed_count = 0
        for candidate in events:
            key = (candidate.owner_id, candidate.session_id)
            async with self._session_lock(key):
                if await self._has_active_work(key):
                    continue
                current = await self.repository.get_event(
                    owner_id=candidate.owner_id,
                    session_id=candidate.session_id,
                    event_id=candidate.id,
                )
                if current is None or current.status != "open":
                    continue
                reason = self.segmenter.due_close_reason(current, now=current_time)
                if reason is None:
                    continue
                await self._close_event(
                    current,
                    ended_at=event_last_evidence_at(current),
                    reason=reason,
                )
                closed_count += 1
        return closed_count

    async def close(self) -> None:
        if self._shutdown_task is None:
            self._closing = True
            self._shutdown_task = asyncio.create_task(self._drain_owned_tasks())
        task = self._shutdown_task
        cancelled = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancelled = error
                if task.cancelled():
                    break
            except Exception:
                break
        try:
            task.result()
        except BaseException:
            if cancelled is not None:
                raise cancelled from None
            raise
        if cancelled is not None:
            raise cancelled

    async def _drain_owned_tasks(self) -> None:
        tasks = list(self._close_tasks.values()) + list(self._tool_attachment_tasks.values())
        self._close_tasks.clear()
        if not tasks:
            return
        try:
            await asyncio.wait(tasks, timeout=SESSION_CLOSE_DRAIN_TIMEOUT_SECONDS)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _attach(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        evidence: EventEvidenceDescriptor,
    ) -> EvidenceAttachmentOutcome:
        key = (owner_id, session_id)
        async with self._session_lock(key):
            current = await self.repository.get_open_event(
                owner_id=owner_id,
                session_id=session_id,
            )
            latest_closed = (
                await self.repository.get_latest_closed_event(
                    owner_id=owner_id,
                    session_id=session_id,
                )
                if current is None or evidence.occurred_at < current.started_at
                else None
            )
            decision = self.segmenter.decide(
                current=current,
                evidence=evidence,
                latest_closed=latest_closed,
            )
            if decision.action == "unattached":
                return EvidenceAttachmentOutcome(False, None, decision.reason)
            if decision.action == "split" and current is not None:
                await self._close_event(
                    current,
                    ended_at=decision.previous_event_ended_at
                    or event_last_evidence_at(current),
                    reason=decision.reason,
                )
                current = None
            if current is None:
                try:
                    current = await self.repository.get_or_create_open_event(
                        owner_id=owner_id,
                        session_id=session_id,
                        event_kind=event_kind_for_modalities((evidence.modality,)),
                        started_at=evidence.occurred_at,
                        modalities=(evidence.modality,),
                    )
                except ExperienceSessionMissingError:
                    return EvidenceAttachmentOutcome(False, None, "session_not_active")

            for _ in range(3):
                aggregate = reduce_event_aggregate(current, evidence)
                try:
                    _, updated, created = (
                        await self.repository.attach_evidence_and_update_aggregate(
                            owner_id=owner_id,
                            session_id=session_id,
                            event_id=current.id,
                            evidence_type=evidence.evidence_type,
                            evidence_ref_id=evidence.evidence_ref_id,
                            source_sequence=evidence.source_sequence,
                            metadata=evidence.metadata,
                            event_kind=aggregate.event_kind,
                            modalities=aggregate.modalities,
                            title=aggregate.title,
                            summary=aggregate.summary,
                            scene=aggregate.scene,
                            activity=aggregate.activity,
                            state=aggregate.state,
                            importance=aggregate.importance,
                            confidence=aggregate.confidence,
                            retention_expires_at=current.retention_expires_at,
                            expected_updated_at=current.updated_at,
                            evidence_occurred_at=evidence.occurred_at,
                        )
                    )
                except ExperienceEventClosedError:
                    return EvidenceAttachmentOutcome(
                        False,
                        None,
                        "event_closed_during_attach",
                    )
                except ExperienceConflictError:
                    refreshed = await self.repository.get_event(
                        owner_id=current.owner_id,
                        session_id=current.session_id,
                        event_id=current.id,
                    )
                    if refreshed is None or refreshed.status != "open":
                        return EvidenceAttachmentOutcome(
                            False,
                            None,
                            "event_closed_during_attach",
                        )
                    current = refreshed
                    refreshed_decision = self.segmenter.decide(
                        current=current, evidence=evidence, latest_closed=latest_closed,
                    )
                    if refreshed_decision.action != "join":
                        return EvidenceAttachmentOutcome(False, None, "event_changed_during_attach")
                    continue
                return EvidenceAttachmentOutcome(
                    True,
                    updated,
                    "attached" if created else "already_attached",
                )
            raise ExperienceConflictError("event aggregate remained contended")

    async def _close_event(
        self,
        event: ExperienceEvent,
        *,
        ended_at: datetime,
        reason: str,
    ) -> ExperienceEvent:
        finalized_at = self._now()
        if self._keyframe_selector is not None and self._keyframe_policy is not None:
            closed = await self.repository.close_event_with_keyframes(
                owner_id=event.owner_id,
                session_id=event.session_id,
                event_id=event.id,
                ended_at=max(event.started_at, ended_at),
                boundary_reason=reason,
                finalized_at=finalized_at,
                candidate_expires_at=finalized_at
                + timedelta(seconds=self._keyframe_policy.candidate_retention_seconds),
                max_count=self._keyframe_policy.consolidation_candidate_max_count,
                selector=lambda candidates: self._keyframe_selector.select(
                    candidates,
                    limit=self._keyframe_policy.consolidation_candidate_max_count,
                ),
            )
        else:
            closed = await self.repository.close_event(
                owner_id=event.owner_id,
                session_id=event.session_id,
                event_id=event.id,
                ended_at=max(event.started_at, ended_at),
                boundary_reason=reason,
                keyframes_finalized_at=finalized_at,
                consolidation_ready_at=finalized_at,
            )
        if event.status == "open" and closed.status != "open":
            await self._publish_closed(closed)
        return closed

    async def _retain_observation_candidates(
        self,
        observation: VisualObservation,
        outcome: EvidenceAttachmentOutcome,
    ) -> None:
        if (
            not outcome.attached
            or outcome.event is None
            or not observation.keyframe_candidates
            or self._keyframe_selector is None
            or self._keyframe_policy is None
        ):
            return
        try:
            await self.repository.reconcile_event_candidates(
                owner_id=observation.owner_id,
                session_id=observation.session_id,
                event_id=outcome.event.id,
                observation_id=observation.id,
                proposed_frame_ids=tuple(
                    item.visual_frame_id for item in observation.keyframe_candidates
                ),
                expires_at=self._now()
                + timedelta(seconds=self._keyframe_policy.candidate_retention_seconds),
                max_count=self._keyframe_policy.open_event_candidate_max_count,
                selector=lambda candidates: self._keyframe_selector.select(
                    candidates,
                    limit=self._keyframe_policy.open_event_candidate_max_count,
                ),
            )
        except ExperienceEventClosedError:
            return
        except Exception:
            LOGGER.exception(
                "failed to retain event keyframe candidates owner=%s session=%s event=%s",
                observation.owner_id,
                observation.session_id,
                outcome.event.id,
            )

    async def _close_after_drain(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        closed_at: datetime,
    ) -> None:
        await self._wait_for_idle(owner_id=owner_id, session_id=session_id)
        key = (owner_id, session_id)
        async with self._session_lock(key):
            event = await self.repository.get_open_event(
                owner_id=owner_id,
                session_id=session_id,
            )
            if event is not None:
                await self._close_event(
                    event,
                    ended_at=max(event_last_evidence_at(event), closed_at),
                    reason="session_closed",
                )

    async def _wait_for_idle(self, *, owner_id: str, session_id: UUID) -> None:
        key = (owner_id, session_id)
        try:
            async with asyncio.timeout(SESSION_CLOSE_DRAIN_TIMEOUT_SECONDS):
                async with self._activity_condition:
                    await self._activity_condition.wait_for(
                        lambda: self._active_counts.get(key, 0) == 0
                    )
        except TimeoutError:
            LOGGER.warning(
                "experience event close drain timed out owner=%s session=%s",
                owner_id,
                session_id,
            )

    def _finish_close_task(
        self,
        key: tuple[str, UUID],
        task: asyncio.Task[None],
    ) -> None:
        self._close_tasks.pop(key, None)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            LOGGER.exception(
                "experience event session-close task failed owner=%s session=%s",
                key[0],
                key[1],
            )

    def _session_lock(self, key: tuple[str, UUID]) -> asyncio.Lock:
        lock = self._session_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[key] = lock
        return lock

    async def _has_active_work(self, key: tuple[str, UUID]) -> bool:
        async with self._activity_condition:
            return self._active_counts.get(key, 0) > 0

    async def _update_working_memory_pointer(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        outcome: EvidenceAttachmentOutcome,
    ) -> None:
        if (
            not outcome.attached
            or outcome.event is None
            or self._working_memory_service is None
        ):
            return
        await self._working_memory_service.set_latest_experience_event_id(
            owner_id=owner_id,
            session_id=session_id,
            event_id=outcome.event.id,
        )

    async def _publish_closed(self, event: ExperienceEvent) -> None:
        if self._domain_event_bus is None or event.ended_at is None:
            return
        await self._domain_event_bus.publish(
            DomainEvent(
                name=DomainEventName.EXPERIENCE_EVENT_CLOSED,
                payload=ExperienceEventClosedPayload(
                    owner_id=event.owner_id,
                    session_id=event.session_id,
                    experience_event_id=event.id,
                    started_at=event.started_at,
                    ended_at=event.ended_at,
                    boundary_reason=event.boundary_reason,
                ),
            )
        )
