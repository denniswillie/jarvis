from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import UUID, uuid4

from app.assistant.repository import BehaviorProfileRepository
from app.experience.service import ExperienceEventService
from app.knowledge.visual_delivery_sources import observation_delivery_source
from app.memory.prospective.context import ProspectiveContextBuilder
from app.memory.prospective.service import ProspectiveMemoryService
from app.realtime.socket_sender import is_closed_websocket_error
from app.runtime.concurrency import ModelCallPriority, PriorityAsyncLimiter
from app.runtime.events import (
    DomainEvent,
    DomainEventBus,
    DomainEventName,
    PerceptionObservationCompletedPayload,
)
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.visual_memory.context import VisualAnalyzerContextBuilder
from app.visual_memory.interrupts import VisualInterruptService
from app.visual_memory.models import VisualFrameRecord
from app.visual_memory.repository import VisualFrameRepository
from app.perception.contracts import (
    FAST_PERCEPTION_CONTRACT_VERSION,
    FastPerceptionFrame,
    FastProspectiveMatch,
    FastPerceptionOutput,
    FastPerceptionRequest,
    FastPerceptionTranscript,
)
from app.perception.fast_analyzer import FastPerceptionAnalyzer
from app.perception.evidence_consumer import PerceptionEvidenceConsumer
from app.perception.live_brief import LivePerceptionBriefBridge
from app.perception.applicability import PerceptionEffectUnavailable
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.perception.models import (
    KeyframeCandidate,
    ObservationEntity,
    ObservationFrameLink,
    ObservationStateChange,
    VisibleTextObservation,
    VisualObservation,
)
from app.perception.repository import PerceptionRepository
from app.perception.services import WorkingMemoryService
from app.perception.session_analysis_coordinator import SessionAnalysisCoordinator


logger = logging.getLogger(__name__)
VisualDiagnosticSender = Callable[[dict[str, object]], Awaitable[None]]


class VisualAnalysisPipeline:
    def __init__(
        self,
        *,
        settings: JarvisSettings,
        repository: VisualFrameRepository,
        session_repository: SessionRepository,
        behavior_repository: BehaviorProfileRepository,
        interrupt_service: VisualInterruptService,
        model_limiter: PriorityAsyncLimiter,
        fast_analyzer: FastPerceptionAnalyzer | None = None,
        perception_repository: PerceptionRepository | None = None,
        sensory_frame_window: RecentVisualFrameWindow | None = None,
        domain_event_bus: DomainEventBus | None = None,
        working_memory_service: WorkingMemoryService | None = None,
        experience_event_service: ExperienceEventService | None = None,
        prospective_context_builder: ProspectiveContextBuilder | None = None,
        prospective_service: ProspectiveMemoryService | None = None,
        live_brief_bridge_for: Callable[[str, UUID], LivePerceptionBriefBridge | None] | None = None,
        perception_evidence_consumer_for: Callable[[str, UUID], PerceptionEvidenceConsumer | None] | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._context_builder = VisualAnalyzerContextBuilder(
            settings=settings,
            visual_repository=repository,
            session_repository=session_repository,
            behavior_repository=behavior_repository,
            working_memory_reader=working_memory_service,
            prospective_context_builder=prospective_context_builder,
        )
        self._interrupt_service = interrupt_service
        self._model_limiter = model_limiter
        self._fast_analyzer = fast_analyzer
        self._perception_repository = perception_repository
        self._observation_publication = None
        self._sensory_frame_window = sensory_frame_window
        self._domain_event_bus = domain_event_bus
        self._working_memory_service = working_memory_service
        self._experience_event_service = experience_event_service
        self._prospective_service = prospective_service
        # Borrowed session-owned bridges; lookup must be synchronous and perform
        # no retrieval. No resolver means the unchanged, default-off path.
        self._live_brief_bridge_for = live_brief_bridge_for
        self._perception_evidence_consumer_for = perception_evidence_consumer_for
        self._perception_evidence_sessions: dict[tuple[str, UUID], Any] = {}
        self._live_briefs: dict[tuple[str, UUID], LivePerceptionBriefBridge] = {}
        self._closed = False
        self._coordinator = SessionAnalysisCoordinator[_QueuedAnalysis](
            self._analyze_queued,
            coalesce_latest=True,
        )

    def set_observation_publication(self, publication) -> None:
        """Configure the selected durable sink before admitting frame analysis."""
        if (self._live_briefs or self._perception_evidence_sessions) and publication is not self._observation_publication:
            raise RuntimeError("Cannot replace canonical capture with active perception sessions")
        self._observation_publication = publication
        if self._perception_repository is not None:
            self._perception_repository.observation_publication = publication
        elif publication is not None and self._fast_analyzer is not None:
            raise RuntimeError("Visual analysis requires its perception repository")

    @property
    def perception_publication(self):
        """Borrow the selected canonical publisher; never create another sink."""
        return self._observation_publication

    @property
    def perception_frame_window(self) -> RecentVisualFrameWindow | None:
        """Borrow the real socket-owned window, including its connection leases."""
        return self._sensory_frame_window

    @property
    def perception_resolution_resources(self):
        """Shared model/limiter/bus; no additional client or full-rate analyzer."""
        model = getattr(self._fast_analyzer, "temporal_model", None)
        if not callable(getattr(model, "generate_temporal_json", None)):
            return None
        return model, self._model_limiter, self._domain_event_bus

    @property
    def perception_focus_service(self):
        """Borrow the existing watch/notification owner, never create another one."""
        return self._prospective_service

    def register_perception_evidence_session(self, session) -> None:
        """Activate an admitted session entry; reject silent owner replacement.

        The entry exposes scope, capture(request), consumer and synchronous,
        idempotent activate()/close(). Application composition creates it.
        """
        if not self.supports_perception_briefs:
            raise RuntimeError("Canonical perception capture is unavailable")
        key = (session.scope.owner_id, session.scope.session_id)
        existing = self._perception_evidence_sessions.get(key)
        if existing is session:
            return
        if existing is not None:
            raise RuntimeError("A perception evidence owner is already registered")
        session.activate()
        self._perception_evidence_sessions[key] = session

    def unregister_perception_evidence_session(self, session) -> None:
        key = (session.scope.owner_id, session.scope.session_id)
        if self._perception_evidence_sessions.get(key) is session:
            del self._perception_evidence_sessions[key]
            session.close()

    @property
    def supports_perception_briefs(self) -> bool:
        return (not self._closed and self._observation_publication is not None
                and self._fast_analyzer is not None and self._perception_repository is not None
                and self._sensory_frame_window is not None)

    def register_live_brief(self, bridge: LivePerceptionBriefBridge) -> None:
        """Borrow an admitted session's bridge; never replace another owner."""
        if not self.supports_perception_briefs or not bridge.enabled:
            raise RuntimeError("Canonical perception capture is unavailable")
        key = (bridge.scope.owner_id, bridge.scope.session_id)
        existing = self._live_briefs.get(key)
        if existing is not None and existing is not bridge:
            raise RuntimeError("A perception brief owner is already registered")
        self._live_briefs[key] = bridge

    def unregister_live_brief(self, bridge: LivePerceptionBriefBridge) -> None:
        key = (bridge.scope.owner_id, bridge.scope.session_id)
        if self._live_briefs.get(key) is bridge:
            del self._live_briefs[key]

    def live_brief_for(self, owner_id: str, session_id: UUID) -> LivePerceptionBriefBridge | None:
        """Synchronous lookup only; frame handling never waits for enrichment."""
        if self._closed:
            return None
        bridge = self._live_briefs.get((owner_id, session_id))
        if bridge is None and self._live_brief_bridge_for is not None:
            bridge = self._live_brief_bridge_for(owner_id, session_id)
        return bridge

    async def close(self) -> None:
        self._closed = True
        evidence_sessions = tuple(self._perception_evidence_sessions.values())
        self._perception_evidence_sessions.clear()
        for session in evidence_sessions:
            session.close()
        self._live_briefs.clear()
        # Session runtime owns bridge shutdown; these are borrowed references.
        await self._coordinator.close()
        for session in evidence_sessions:
            if join := getattr(session, "aclose", None):
                await join()

    async def drain(self) -> None:
        """Wait until all currently queued frame analyses have settled."""
        await self._coordinator.drain()

    async def pending_count(self, session_id: UUID) -> int:
        """Return the current per-session analyzer backlog for diagnostics/evals."""
        return await self._coordinator.pending_count(session_id)

    async def enqueue(
        self,
        *,
        frame: VisualFrameRecord,
        image_bytes: bytes,
        send_diagnostic: VisualDiagnosticSender | None = None,
    ) -> None:
        if self._fast_analyzer is None:
            await self._repository.mark_analysis_skipped(
                owner_id=frame.owner_id,
                frame_id=frame.id,
                reason="gemini_visual_analyzer_not_configured",
            )
            await _best_effort_send(
                send_diagnostic,
                {
                    "type": "visual.frame.analysis_skipped",
                    "frame_id": str(frame.id),
                    "frame_sequence": frame.frame_sequence,
                    "code": "gemini_visual_analyzer_not_configured",
                },
            )
            return

        await self._repository.mark_analysis_pending(owner_id=frame.owner_id, frame_id=frame.id)
        queued = _QueuedAnalysis(
            frame=frame,
            image_bytes=bytes(image_bytes),
            send_diagnostic=send_diagnostic,
        )
        replaced = await self._coordinator.submit(frame.session_id, queued)
        if replaced is not None:
            await self._repository.mark_analysis_skipped(
                owner_id=replaced.frame.owner_id,
                frame_id=replaced.frame.id,
                reason="visual_analyzer_latest_frame_replaced",
            )
            await _best_effort_send(
                replaced.send_diagnostic,
                {
                    "type": "visual.frame.analysis_skipped",
                    "frame_id": str(replaced.frame.id),
                    "frame_sequence": replaced.frame.frame_sequence,
                    "code": "visual_analyzer_latest_frame_replaced",
                },
            )

    async def _analyze_queued(self, queued: "_QueuedAnalysis") -> None:
        if self._experience_event_service is None:
            await self._analyze_queued_inner(queued)
            return

        async with self._experience_event_service.activity(
            owner_id=queued.frame.owner_id,
            session_id=queued.frame.session_id,
        ):
            await self._analyze_queued_inner(queued)

    async def _analyze_queued_inner(self, queued: "_QueuedAnalysis") -> None:
        frame = queued.frame
        analysis_input = None
        observation_stored = False
        try:
            context = await self._context_builder.build(frame)
            if context is None:
                await self._repository.mark_analysis_failed(
                    owner_id=frame.owner_id,
                    frame_id=frame.id,
                )
                return
            if (
                self._fast_analyzer is None
                or self._perception_repository is None
                or self._sensory_frame_window is None
            ):
                raise RuntimeError("temporal perception runtime is not configured")

            request = self._build_fast_request(
                frame=frame,
                image_bytes=queued.image_bytes,
                context=context,
            )
            bridge = self.live_brief_for(frame.owner_id, frame.session_id)
            if bridge is not None:
                request = bridge.attach(request)
            # Capture socket ownership before input fingerprinting/model work.
            # Keep THIS entry/consumer across all awaits: a newly registered
            # session must never receive the previous owner's analysis result.
            evidence_session = self._perception_evidence_sessions.get((frame.owner_id, frame.session_id))
            evidence_consumer = None
            if evidence_session is not None:
                request = evidence_session.capture(request)
                evidence_consumer = evidence_session.consumer
            elif self._perception_evidence_consumer_for is not None and request.perception_brief is not None:
                try:
                    evidence_consumer = self._perception_evidence_consumer_for(frame.owner_id, frame.session_id)
                except Exception as error:
                    logger.warning("perception evidence lookup unavailable error_type=%s", type(error).__name__)
            if request.perception_brief is not None and self._observation_publication is None:
                raise RuntimeError("Perception assignments require canonical observation input capture")
            if self._observation_publication is not None:
                request, analysis_input = await self._observation_publication.prepare_inputs(request, context)
            async with self._model_limiter.limit(ModelCallPriority.NORMAL):
                if analysis_input is not None:
                    await self._observation_publication.validate_input(request, analysis_input)
                async with asyncio.timeout(
                    self._settings.VISUAL_FAST_PERCEPTION_TIMEOUT_SECONDS
                ):
                    output = await self._fast_analyzer.analyze(request)
            observation = _observation_from_fast_output(
                request=request,
                output=output,
                provider=self._fast_analyzer.provider,
                model=self._fast_analyzer.model,
            )
            links = _frame_links(request, output.keyframe_candidates)
            await self._perception_repository.insert_observation(
                observation=observation,
                frame_links=links,
                **({"analysis_input": analysis_input} if analysis_input is not None else {}),
            )
            observation_stored = True
            await self._attach_observation_to_experience(observation)
            source_payload_id = None
            effect_guard = None
            if request.perception_brief is not None:
                source_payload_id = await observation_delivery_source(
                    self._observation_publication, owner_id=frame.owner_id,
                    session_id=frame.session_id, observation_id=observation.id)
                effect_guard = bridge.effect_guard_for(request, source_payload_id)
            try:
                if effect_guard is not None:
                    async with effect_guard():
                        pass
                if self._working_memory_service is not None:
                    await self._working_memory_service.apply_observation(observation,
                        **({"effect_guard": effect_guard} if effect_guard is not None else {}))
            except PerceptionEffectUnavailable:
                # The original is valid historical evidence, not a failed
                # upload/model call and not a current goal resolution.
                await self._repository.apply_analysis_result(owner_id=frame.owner_id,
                    frame_id=frame.id, frame_description=None, memory_tags=[])
                await _best_effort_send(queued.send_diagnostic, {
                    "type": "visual.frame.analyzed", "frame_id": str(frame.id),
                    "frame_sequence": frame.frame_sequence,
                    "perception_observation_id": str(observation.id),
                    "current_applicability": "historical_only",
                    "code": "perception_assignment_unavailable",
                })
                return
            if source_payload_id is None and self._observation_publication is not None and (
                output.interrupt_candidate is not None or output.prospective_matches
            ):
                source_payload_id = await observation_delivery_source(
                    self._observation_publication, owner_id=frame.owner_id,
                    session_id=frame.session_id, observation_id=observation.id)
            await self._consume_perception_findings_best_effort(
                request=request, output=output, observation=observation,
                analysis_input=analysis_input, effect_guard=effect_guard,
                send_diagnostic=queued.send_diagnostic,
                consumer=evidence_consumer,
            )
            await self._apply_completed_result(
                frame=frame,
                output=output,
                send_diagnostic=queued.send_diagnostic,
                observation_id=observation.id,
                source_payload_id=source_payload_id,
                effect_guard=effect_guard,
            )
            await self._publish_observation_completed(observation)
            await self._process_prospective_matches_best_effort(
                request=request,
                matches=output.prospective_matches,
                frame_id=frame.id,
                source_payload_id=source_payload_id,
                effect_guard=effect_guard,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("visual temporal perception failed for frame %s", frame.id)
            try:
                await self._repository.mark_analysis_failed(
                    owner_id=frame.owner_id,
                    frame_id=frame.id,
                )
            except Exception:
                logger.exception("failed to persist analysis failure for frame %s", frame.id)
            await _best_effort_send(
                queued.send_diagnostic,
                {
                    "type": "visual.frame.analysis_failed",
                    "frame_id": str(frame.id),
                    "frame_sequence": frame.frame_sequence,
                    "code": "visual_temporal_perception_failed",
                },
            )

        finally:
            if analysis_input is not None and not observation_stored:
                try:
                    await self._observation_publication.discard_input(analysis_input)
                except Exception:
                    logger.warning("unused visual input cleanup failed frame=%s", frame.id, exc_info=True)

    async def _consume_perception_findings_best_effort(
        self, *, request, output, observation, analysis_input, effect_guard, send_diagnostic,
        consumer=None,
    ) -> None:
        # Default-off and borrowed per-session ownership, like the brief bridge.
        # Optional evidence failure cannot discard the usable scene or authorize
        # a second notification route. Cancellation still propagates normally.
        if consumer is None or request.perception_brief is None or not output.perception_findings:
            return
        try:
            result = await consumer.consume(request=request, output=output, observation=observation,
                analysis_input=analysis_input, effect_guard=effect_guard)
            await _best_effort_send(send_diagnostic, {
                "type": "visual.perception_evidence", "perception_observation_id": str(observation.id),
                "status": result.status, "code": result.reason,
                "evidence_event_id": str(result.record.id) if result.record is not None else None,
                "owner_effect": "none",
            })
        except Exception as error:
            logger.warning("perception evidence consumer unavailable error_type=%s", type(error).__name__)

    async def _process_prospective_matches_best_effort(
        self,
        *,
        request: FastPerceptionRequest,
        matches: list[FastProspectiveMatch],
        frame_id: UUID,
        source_payload_id: UUID | None = None,
        effect_guard=None,
    ) -> None:
        if self._prospective_service is None or not matches:
            return
        try:
            await self._prospective_service.process_matches(
                request=request,
                matches=matches,
                **({"source_payload_id": source_payload_id} if source_payload_id is not None else {}),
                **({"effect_guard": effect_guard} if effect_guard is not None else {}),
            )
        except Exception:
            logger.warning(
                "prospective delivery failed for completed frame %s",
                frame_id,
                exc_info=True,
            )

    def _build_fast_request(self, *, frame, image_bytes, context) -> FastPerceptionRequest:
        assert self._sensory_frame_window is not None
        frames = [
            item
            for item in self._sensory_frame_window.window(
                frame.owner_id,
                frame.session_id,
                max_frames=self._settings.VISUAL_FAST_PERCEPTION_MAX_WINDOW_FRAMES,
                max_age_seconds=self._settings.VISUAL_SENSORY_WINDOW_TTL_SECONDS,
            )
            if item.frame_sequence <= frame.frame_sequence
            and item.captured_at <= frame.captured_at
        ]
        if not any(item.frame_id == frame.id for item in frames):
            frames.append(_sensory_frame_from_record(frame, image_bytes))
        frames.sort(key=lambda item: (item.captured_at, item.frame_sequence))
        frames = frames[-self._settings.VISUAL_FAST_PERCEPTION_MAX_WINDOW_FRAMES :]
        primary = next(item for item in frames if item.frame_id == frame.id)
        frames = [item for item in frames if item.frame_id != frame.id] + [primary]
        return FastPerceptionRequest(
            owner_id=frame.owner_id,
            session_id=frame.session_id,
            window_id=uuid4(),
            frames=tuple(
                FastPerceptionFrame(
                    frame_id=item.frame_id,
                    frame_sequence=item.frame_sequence,
                    captured_at=item.captured_at,
                    source=item.source,
                    content_type=item.content_type,
                    image_bytes=item.image_bytes,
                )
                for item in frames
            ),
            previous_working_state=context.previous_working_state,
            existing_entity_ids=_existing_entity_ids(context.previous_working_state),
            recent_audio_transcripts=tuple(
                FastPerceptionTranscript(
                    role=message.role,
                    content=message.content,
                    created_at=message.created_at,
                )
                for message in context.recent_audio_transcripts
            ),
            behavior_instructions=context.behavior_instructions,
            active_reminders=tuple(
                {
                    "id": str(reminder.id),
                    "title": reminder.title,
                    "details": reminder.details,
                    "remind_at": reminder.remind_at.isoformat()
                    if reminder.remind_at
                    else None,
                    "condition": reminder.condition,
                }
                for reminder in context.active_reminders
            ),
            active_focus=context.active_focus,
            recent_interrupts=tuple(
                {
                    "message": item.message,
                    "reason": item.reason,
                    "confidence": item.confidence,
                    "created_at": item.created_at.isoformat(),
                }
                for item in context.recent_visual_interrupts
            ),
            requested_at=context.current_utc_time,
        )

    async def _apply_completed_result(
        self,
        *,
        frame: VisualFrameRecord,
        output: FastPerceptionOutput,
        send_diagnostic: VisualDiagnosticSender | None,
        observation_id: UUID,
        source_payload_id: UUID | None = None,
        effect_guard=None,
    ) -> None:
        await self._repository.apply_analysis_result(
            owner_id=frame.owner_id,
            frame_id=frame.id,
            # The selected source already owns the analyzer original. Do not
            # leave an unbound duplicate that can race source invalidation.
            frame_description=output.summary if self._observation_publication is None else None,
            memory_tags=list(output.memory_tags) if self._observation_publication is None else [],
        )
        interrupt = output.interrupt_candidate
        # A tracked match owns this analysis's speech, including a rejected
        # attempt. Never bypass its completion bookkeeping through the generic
        # route; later evidence can retry the still-active tracked search.
        tracked_delivery = any(
            match.confidence >= self._settings.VISUAL_INTERRUPT_CONFIDENCE_THRESHOLD
            for match in output.prospective_matches
        )
        if (
            interrupt is not None
            and not tracked_delivery
            and interrupt.confidence
            >= self._settings.VISUAL_INTERRUPT_CONFIDENCE_THRESHOLD
        ):
            try:
                await self._interrupt_service.maybe_interrupt(
                    owner_id=frame.owner_id,
                    session_id=frame.session_id,
                    audio_session_id=frame.audio_session_id,
                    visual_frame_id=frame.id,
                    source_received_at=frame.server_received_at,
                    source_captured_at=frame.captured_at,
                    message=interrupt.message,
                    reason=interrupt.reason,
                    confidence=interrupt.confidence,
                    **({"source_payload_id": source_payload_id} if source_payload_id is not None else {}),
                    **({"effect_guard": effect_guard} if effect_guard is not None else {}),
                )
            except Exception:
                logger.info(
                    "visual interrupt handling failed for frame %s",
                    frame.id,
                    exc_info=True,
                )
        await _best_effort_send(
            send_diagnostic,
            {
                "type": "visual.frame.analyzed",
                "frame_id": str(frame.id),
                "frame_sequence": frame.frame_sequence,
                "analyzed_at": datetime.now(UTC).isoformat(),
                "perception_observation_id": str(observation_id),
            },
        )

    async def _publish_observation_completed(self, observation: VisualObservation) -> None:
        if self._domain_event_bus is None:
            return
        await self._domain_event_bus.publish(
            DomainEvent(
                name=DomainEventName.PERCEPTION_OBSERVATION_COMPLETED,
                payload=PerceptionObservationCompletedPayload(
                    owner_id=observation.owner_id,
                    session_id=observation.session_id,
                    observation_id=observation.id,
                    frame_id=observation.primary_visual_frame_id,
                ),
            )
        )

    async def _attach_observation_to_experience(
        self,
        observation: VisualObservation,
    ) -> None:
        if self._experience_event_service is None:
            return
        try:
            await self._experience_event_service.attach_observation(observation)
        except Exception:
            logger.exception(
                "failed to attach visual observation to experience event",
                extra={
                    "observation_id": str(observation.id),
                    "session_id": str(observation.session_id),
                },
            )


class _QueuedAnalysis:
    def __init__(self, *, frame, image_bytes, send_diagnostic) -> None:
        self.frame: VisualFrameRecord = frame
        self.image_bytes: bytes = image_bytes
        self.send_diagnostic: VisualDiagnosticSender | None = send_diagnostic


def _sensory_frame_from_record(
    frame: VisualFrameRecord,
    image_bytes: bytes,
) -> SensoryVisualFrame:
    return SensoryVisualFrame(
        frame_id=frame.id,
        owner_id=frame.owner_id,
        session_id=frame.session_id,
        audio_session_id=frame.audio_session_id,
        visual_session_id=frame.visual_session_id,
        frame_sequence=frame.frame_sequence,
        source=frame.source,
        captured_at=frame.captured_at,
        server_received_at=frame.server_received_at,
        content_type=frame.content_type,
        width=frame.width,
        height=frame.height,
        storage_bucket=frame.storage_bucket,
        storage_path=frame.storage_path,
        raw_image_expires_at=frame.raw_image_expires_at,
        image_bytes=bytes(image_bytes),
    )


def _existing_entity_ids(state: Mapping[str, Any]) -> frozenset[str]:
    raw_entities = state.get("entities")
    if isinstance(raw_entities, Mapping):
        values = raw_entities.values()
    elif isinstance(raw_entities, list):
        values = raw_entities
    else:
        return frozenset()
    identifiers: set[str] = set()
    for value in values:
        if not isinstance(value, Mapping):
            continue
        identifier = value.get("entity_id") or value.get("id")
        if isinstance(identifier, str) and identifier.strip():
            identifiers.add(identifier.strip())
    return frozenset(identifiers)


def _observation_from_fast_output(*, request, output, provider, model) -> VisualObservation:
    now = datetime.now(UTC)
    return VisualObservation(
        id=uuid4(),
        owner_id=request.owner_id,
        session_id=request.session_id,
        source=request.primary_frame.source,
        started_at=request.frames[0].captured_at,
        ended_at=request.primary_frame.captured_at,
        primary_visual_frame_id=request.primary_frame.frame_id,
        summary=output.summary,
        scene=output.scene.label if output.scene else None,
        activity=output.activity.label if output.activity else None,
        scene_confidence=output.scene.confidence if output.scene else None,
        activity_confidence=output.activity.confidence if output.activity else None,
        entities=tuple(
            ObservationEntity(
                name=entity.name,
                category=entity.type,
                confidence=entity.confidence,
                attributes={
                    **entity.attributes,
                    "entity_id": entity.existing_entity_id,
                    "local_key": entity.new_local_key,
                    "state": entity.state,
                    "last_visible_frame_id": str(entity.last_visible_frame_id),
                },
            )
            for entity in output.entities
        ),
        visible_text=tuple(
            VisibleTextObservation(
                text=item.text,
                context=item.context,
                confidence=item.confidence,
            )
            for item in output.visible_text
        ),
        state_changes=tuple(
            ObservationStateChange(
                subject=item.subject_local_key,
                attribute="state",
                before=None,
                after=item.change,
                confidence=item.confidence,
            )
            for item in output.state_changes
        ),
        memory_tags=tuple(output.memory_tags),
        keyframe_candidates=tuple(
            KeyframeCandidate(
                visual_frame_id=item.frame_id,
                reason=item.reason,
                confidence=item.score,
            )
            for item in output.keyframe_candidates
        ),
        interrupt_candidate=(
            output.interrupt_candidate.model_dump()
            if output.interrupt_candidate is not None
            else None
        ),
        starts_new_event=output.starts_new_event,
        boundary_reason=output.boundary_reason,
        confidence=output.confidence,
        analyzer_provider=provider,
        analyzer_model=model,
        analyzer_contract_version=FAST_PERCEPTION_CONTRACT_VERSION,
        compacted_at=None,
        created_at=now,
    )


def _frame_links(request, keyframe_candidates) -> tuple[ObservationFrameLink, ...]:
    keyframe_ids = {item.frame_id for item in keyframe_candidates}
    return tuple(
        ObservationFrameLink(
            visual_frame_id=frame.frame_id,
            ordinal=index,
            role=(
                "primary"
                if frame.frame_id == request.primary_frame.frame_id
                else "key_candidate"
                if frame.frame_id in keyframe_ids
                else "context"
            ),
        )
        for index, frame in enumerate(request.frames)
    )


async def _best_effort_send(send_diagnostic: VisualDiagnosticSender | None, payload: dict[str, object]) -> None:
    if send_diagnostic is None:
        return
    try:
        await send_diagnostic(payload)
    except Exception as error:
        if not is_closed_websocket_error(error):
            # Diagnostics never own completed observations. Preserve the error
            # type, but not a transport exception that can embed media payloads.
            logger.warning("visual diagnostic send failed error_type=%s", type(error).__name__)
