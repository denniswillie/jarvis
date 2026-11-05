from __future__ import annotations

from dataclasses import dataclass, field

import asyncpg

from app.assistant.repository import BehaviorProfileRepository
from app.experience.service import ExperienceEventService
from app.memory.prospective.context import ProspectiveContextBuilder
from app.memory.prospective.service import ProspectiveMemoryService
from app.model_providers.gemini_fast_visual import GeminiFastVisualModel
from app.model_providers.protocols import VisualInspectionModel
from app.perception.fast_analyzer import FastPerceptionAnalyzer
from app.perception.frame_window import RecentVisualFrameWindow
from app.perception.repository import PerceptionRepository
from app.perception.services import WorkingMemoryService
from app.perception.working_memory import WorkingMemoryPolicy
from app.realtime.session_manager import RealtimeSessionManager
from app.runtime.concurrency import PriorityAsyncLimiter
from app.runtime.events import DomainEventBus
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.visual_memory.gemini_client import (
    GeminiVisualAnalyzerClient,
    gemini_visual_config_from_settings,
)
from app.visual_memory.interrupts import VisualInterruptService
from app.visual_memory.pipeline import VisualAnalysisPipeline
from app.visual_memory.repository import VisualFrameRepository


@dataclass
class VisualMemoryServices:
    repository: VisualFrameRepository
    pipeline: VisualAnalysisPipeline
    gemini_client: VisualInspectionModel | None
    working_memory_service: WorkingMemoryService
    companion_runtime: object | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    async def enqueue(self, *, frame, image_bytes, send_diagnostic=None) -> None:
        """Exactly one selected analyzer, after router storage/camera validation."""
        if self._closed:
            raise RuntimeError("visual services are closed")
        if self.companion_runtime is not None:
            await self.companion_runtime.enqueue_frame(frame, image_bytes)
            return
        await self.pipeline.enqueue(frame=frame, image_bytes=image_bytes,
                                    send_diagnostic=send_diagnostic)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.companion_runtime is not None:
            await self.companion_runtime.close()
        await self.pipeline.close()
        if self.gemini_client is not None:
            await self.gemini_client.close()


def create_visual_memory_services(
    *,
    settings: JarvisSettings,
    pool: asyncpg.Pool,
    realtime_session_manager: RealtimeSessionManager,
    model_limiter: PriorityAsyncLimiter,
    sensory_frame_window: RecentVisualFrameWindow,
    domain_event_bus: DomainEventBus,
    working_memory_service: WorkingMemoryService | None = None,
    experience_event_service: ExperienceEventService | None = None,
    prospective_context_builder: ProspectiveContextBuilder | None = None,
    prospective_service: ProspectiveMemoryService | None = None,
) -> VisualMemoryServices:
    repository = VisualFrameRepository(pool)
    session_repository = SessionRepository(pool)
    gemini_config = gemini_visual_config_from_settings(settings)
    gemini_client = (
        GeminiVisualAnalyzerClient(gemini_config)
        if gemini_config is not None
        else None
    )
    perception_repository = PerceptionRepository(pool)
    working_memory_service = working_memory_service or WorkingMemoryService(
        repository=perception_repository,
        policy=WorkingMemoryPolicy(
            ttl_seconds=settings.WORKING_MEMORY_TTL_SECONDS,
            entity_ttl_seconds=settings.WORKING_MEMORY_ENTITY_TTL_SECONDS,
            visible_text_ttl_seconds=settings.WORKING_MEMORY_VISIBLE_TEXT_TTL_SECONDS,
            max_entities=settings.WORKING_MEMORY_MAX_ENTITIES,
            max_recent_changes=settings.WORKING_MEMORY_MAX_RECENT_CHANGES,
        ),
        domain_event_bus=domain_event_bus,
    )
    fast_analyzer = (
        FastPerceptionAnalyzer(GeminiFastVisualModel(gemini_client))
        if gemini_client is not None and settings.VISUAL_FAST_PERCEPTION_ENABLED
        else None
    )
    pipeline = VisualAnalysisPipeline(
        settings=settings,
        repository=repository,
        session_repository=session_repository,
        behavior_repository=BehaviorProfileRepository(pool),
        interrupt_service=VisualInterruptService(
            repository=repository,
            realtime_session_manager=realtime_session_manager,
            dedupe_seconds=settings.VISUAL_INTERRUPT_DEDUPE_SECONDS,
            max_age_seconds=settings.PROSPECTIVE_MAX_EVIDENCE_AGE_SECONDS,
        ),
        model_limiter=model_limiter,
        fast_analyzer=fast_analyzer,
        perception_repository=perception_repository,
        sensory_frame_window=sensory_frame_window,
        domain_event_bus=domain_event_bus,
        working_memory_service=working_memory_service,
        experience_event_service=experience_event_service,
        prospective_context_builder=prospective_context_builder,
        prospective_service=prospective_service,
    )
    return VisualMemoryServices(
        repository=repository,
        pipeline=pipeline,
        gemini_client=gemini_client,
        working_memory_service=working_memory_service,
    )
