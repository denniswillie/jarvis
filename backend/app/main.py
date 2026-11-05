from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.assistant.context_providers import (
    ActiveRemindersContextProvider,
    AssistantContextProvider,
    BehaviorContextProvider,
    CurrentTimeContextProvider,
    PreferencesContextProvider,
    RecentMessagesContextProvider,
    RecentVisualInterruptsContextProvider,
    SessionContextProvider,
)
from app.assistant.memory_context import WorkingMemoryContextProvider
from app.assistant.repository import BehaviorProfileRepository
from app.assistant.profile_router import router as behavior_profile_router
from app.assistant.profile_runtime import BehaviorProfileUpdates
from app.assistant.profile_service import BehaviorProfileService
from app.plugins.router import router as plugins_router
from app.plugins.runtime import initialize_plugin_runtime
from app.assistant.tools import BehaviorToolModule
from app.calendar.google_oauth import GoogleCredentialService
from app.calendar.repository import GoogleOAuthTokenRepository
from app.calendar.router import router as google_calendar_router
from app.calendar.tools import CalendarToolModule, GoogleCalendarClient
from app.db import create_pool
from app.diagnostics.router import router as diagnostics_router
from app.errors import JarvisHTTPError
from app.experience.keyframes import (
    DeterministicKeyframeSelector,
    KeyframeRetentionPolicy,
)
from app.experience.repository import ExperienceRepository
from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.experience.sweeper import (
    ExperienceEventSweeper,
    run_experience_event_sweeper_loop,
)
from app.files.router import router as files_router
from app.files.recording_router import router as recording_router
from app.recordings.runtime import initialize_recording_runtime
from app.files.repository import JarvisFileRepository
from app.files.storage import JarvisFileStorageClient
from app.files.tools import FileToolModule
from app.health import router as health_router
from app.hermes_sessions.background_events import background_wake_router
from app.logging_config import configure_logging
from app.memory.jobs.reconciler import MemoryJobReconciler
from app.memory.jobs.registry import MemoryJobHandlerRegistry
from app.memory.jobs.repository import MemoryJobRepository
from app.memory.jobs.runner import MemoryJobRunner, MemoryJobRunnerConfig
from app.memory.jobs.scheduler import (
    MemoryMaintenanceScheduler,
    run_memory_maintenance_scheduler_loop,
)
from app.memory.retention import (
    DATABASE_MAINTENANCE_JOB_TYPES,
    MemoryMaintenanceJobHandler,
    MemoryRetentionPolicy,
    MemoryRetentionRepository,
)
from app.memory.embeddings import ExperienceEventEmbeddingService
from app.memory.embeddings.job_handlers import (
    EmbedExperienceEventJobHandler,
    EmbedMemoryEntityJobHandler,
    EmbedMemoryItemJobHandler,
)
from app.memory.consolidation.job_handler import (
    ConsolidateExperienceEventJobHandler,
    ExperienceConsolidationRecoveryRule,
)
from app.memory.consolidation.openai_client import OpenAIStructuredOutputClient
from app.memory.consolidation.service import ExperienceConsolidationService
from app.memory.episodic.job_handler import PromoteExperienceEventJobHandler
from app.memory.episodic.repository import EpisodicMemoryRepository
from app.memory.episodic.service import EpisodePromotionService
from app.memory.entities.repository import EntityRepository
from app.memory.forgetting import MemoryForgettingService
from app.memory.repository import MemoryRepository
from app.memory.projection import (
    MemoryProfileRecoveryRule,
    MemoryProjectionLimits,
    MemoryProjectionRepository,
    MemoryProjectionService,
    RefreshMemoryProfileJobHandler,
)
from app.memory.reflection.job_handler import (
    GenerateReflectionJobHandler,
    ReflectionRecoveryRule,
)
from app.memory.reflection.repository import ReflectionRepository
from app.memory.reflection.service import (
    ReflectionGenerationService,
    ReflectionTriggerService,
)
from app.memory.retrieval.evidence_inspector import (
    HistoricalEvidenceInspector,
    HistoricalEvidenceRepository,
)
from app.memory.retrieval.candidates import RetrievalCandidateCollector
from app.memory.retrieval.service import HierarchicalMemoryRetrievalService
from app.memory.retrieval.image_evidence import HistoricalImageService
from app.memory.router import router as memory_router
from app.memory.service import MemoryApplicationService
from app.memory.tools import MemoryToolModule
from app.memory.prospective import (
    ProspectiveContextBuilder,
    ProspectiveContextProvider,
    ProspectiveMemoryService,
    ProspectiveRepository,
    ProspectiveToolModule,
)
from app.memory.semantic.job_handler import (
    ReconcileSemanticMemoryJobHandler,
    SemanticMemoryRecoveryRule,
)
from app.memory.semantic.reconciler import SemanticMemoryReconciler
from app.memory.semantic.repository import SemanticMemoryRepository
from app.model_providers.gemini_multimodal_embeddings import (
    GeminiMultimodalEmbeddingModel,
    GeminiMultimodalEmbeddingNotConfiguredError,
    gemini_multimodal_embedding_config_from_settings,
)
from app.model_providers.gemini_fast_visual import GeminiFastVisualModel
from app.perception.current_image import CurrentImageService
from app.perception.frame_window import RecentVisualFrameWindow
from app.perception.repository import PerceptionRepository
from app.perception.services import WorkingMemoryService
from app.perception.tools import CurrentImageToolModule
from app.perception.working_memory import WorkingMemoryPolicy
from app.preferences.repository import PreferenceRepository
from app.preferences.tools import PreferenceToolModule
from app.realtime.router import router as realtime_router
from app.realtime.session_manager import RealtimeSessionManager
from app.reminders.delivery import ReminderDeliveryService, ReminderMessageGenerator
from app.reminders.repository import ReminderRepository
from app.reminders.router import router as reminders_router
from app.reminders.scheduler import ReminderScheduler, run_reminder_scheduler_loop
from app.reminders.tools import ReminderToolModule
from app.runtime import JarvisRuntimeServices
from app.runtime.concurrency import PriorityAsyncLimiter
from app.runtime.events import DomainEventBus, DomainEventName
from app.sessions.repository import SessionRepository
from app.sessions.router import router as sessions_router
from app.settings import get_settings, is_configured_secret
from app.storage.supabase_config import get_supabase_storage_config
from app.tools.module import ToolModule
from app.tasks.router import router as tasks_router
from app.tasks.service import initialize_task_runtime
from app.attention.service import initialize_attention_runtime
from app.attention.router import router as attention_router
from app.knowledge.runtime import initialize_knowledge_runtime
from app.knowledge.selection import configure_selected_memory
from app.projects.router import router as projects_router
from app.runtime.feature_config import MemoryAuthority, get_second_brain_settings
from app.visual_memory.latest_frame_buffer import LatestVisualFrameBuffer
from app.visual_memory.repository import VisualFrameRepository
from app.visual_memory.retention import (
    VisualFrameRetentionService,
    run_visual_frame_retention_loop,
)
from app.visual_memory.router import router as visual_memory_router
from app.visual_memory.services import (
    VisualMemoryServices,
    create_visual_memory_services,
)
from app.visual_memory.gemini_client import GeminiVisualAnalyzerClient
from app.visual_memory.storage import VisualFrameStorageClient
from app.visual_memory.tools import VisualCaptureToolModule
from app.web.fetch import WebFetchCacheRepository, WebFetcher
from app.web.search import OpenAIWebSearch
from app.web.tools import WebToolModule
from app.workspace.router import router as workspace_router
from app.sandbox.view_router import router as sandbox_view_router

logger = logging.getLogger(__name__)


def _create_assistant_context_providers(
    services: JarvisRuntimeServices,
) -> tuple[AssistantContextProvider, ...]:
    pool = services.db_pool
    visual_services = services.visual_memory_services
    visual_repository = (
        visual_services.repository
        if isinstance(visual_services, VisualMemoryServices)
        else VisualFrameRepository(pool) if pool is not None else None
    )
    providers: list[AssistantContextProvider] = [
        CurrentTimeContextProvider(),
        SessionContextProvider(),
        BehaviorContextProvider(
            (getattr(getattr(services, "behavior_profile_service", None), "repository", None)
             or BehaviorProfileRepository(pool)) if pool is not None else None
        ),
        RecentMessagesContextProvider(
            max_items=services.settings.JARVIS_REALTIME_RECENT_MESSAGES_LIMIT
        ),
        ActiveRemindersContextProvider(
            ReminderRepository(pool) if pool is not None else None
        ),
        PreferencesContextProvider(
            PreferenceRepository(pool) if pool is not None else None
        ),
        WorkingMemoryContextProvider(service=services.working_memory_service),
        RecentVisualInterruptsContextProvider(repository=visual_repository),
    ]
    if services.prospective_context_builder is not None:
        providers.append(
            ProspectiveContextProvider(services.prospective_context_builder)
        )
    return tuple(providers)


def _create_tool_modules(services: JarvisRuntimeServices) -> tuple[ToolModule, ...]:
    pool = services.db_pool
    if pool is None:
        return ()

    settings = services.settings
    file_repository = JarvisFileRepository(pool)
    visual_services = services.visual_memory_services
    visual_repository = (
        visual_services.repository
        if isinstance(visual_services, VisualMemoryServices)
        else VisualFrameRepository(pool)
    )
    modules: list[ToolModule] = [
        BehaviorToolModule(getattr(getattr(services, "behavior_profile_service", None), "repository", None)
                           or BehaviorProfileRepository(pool)),
        *([MemoryToolModule(
            repository=MemoryRepository(pool),
            entity_repository=EntityRepository(pool),
            service=services.memory_service,
            forgetting_service=services.memory_forgetting_service,
            projection_service=services.memory_projection_service,
            image_service=services.historical_image_service,
        )] if services.memory_authority == MemoryAuthority.LEGACY else []),
        ReminderToolModule(ReminderRepository(pool)),
        PreferenceToolModule(PreferenceRepository(pool)),
        CalendarToolModule(
            GoogleCalendarClient(
                credentials=GoogleCredentialService(
                    settings=settings,
                    repository=GoogleOAuthTokenRepository(pool),
                ),
                default_timezone=settings.JARVIS_TIMEZONE,
            )
        ),
        WebToolModule(
            fetcher=WebFetcher(
                settings=settings,
                repository=WebFetchCacheRepository(pool),
            ),
            search=OpenAIWebSearch(settings=settings),
        ),
    ]
    if services.jarvis_file_storage is not None:
        modules.append(
            FileToolModule(
                repository=file_repository,
                storage=services.jarvis_file_storage,
            )
        )
    modules.append(
        VisualCaptureToolModule(
            repository=visual_repository,
            latest_frame_buffer=services.latest_visual_frame_buffer,
            file_repository=(
                file_repository if services.jarvis_file_storage is not None else None
            ),
            file_storage=services.jarvis_file_storage,
        )
    )
    if services.current_image_service is not None:
        modules.append(CurrentImageToolModule(services.current_image_service))
    if services.prospective_service is not None:
        modules.append(ProspectiveToolModule(services.prospective_service))
    return tuple(modules)


async def _initialize_runtime_services(services: JarvisRuntimeServices) -> None:
    settings = services.settings
    services.memory_authority = get_second_brain_settings().selected_memory_authority()
    legacy_memory = services.memory_authority == MemoryAuthority.LEGACY
    if is_configured_secret(settings.SUPABASE_DB_URL):
        services.db_pool = await create_pool(settings)
        logger.info("database pool initialized")
        services.behavior_profile_updates = BehaviorProfileUpdates(services.realtime_session_manager)
        services.behavior_profile_service = BehaviorProfileService(BehaviorProfileRepository(
            services.db_pool, on_committed=services.behavior_profile_updates.submit))
        memory_job_repository = None
        memory_job_registry = None
        memory_job_reconciler = None
        maintenance_job_intervals: dict[str, int] = {}
        visual_frame_retention_service = None
        if settings.MEMORY_JOB_RUNNER_ENABLED and legacy_memory:
            memory_job_repository = MemoryJobRepository(
                services.db_pool,
                default_max_attempts=settings.MEMORY_JOB_DEFAULT_MAX_ATTEMPTS,
            )
            memory_job_registry = MemoryJobHandlerRegistry()
            memory_job_reconciler = MemoryJobReconciler(
                pool=services.db_pool,
                repository=memory_job_repository,
            )
        consolidation_enabled = bool(
            memory_job_repository is not None
            and memory_job_registry is not None
            and memory_job_reconciler is not None
            and is_configured_secret(settings.OPENAI_API_KEY)
            and is_configured_secret(settings.OPENAI_CONSOLIDATION_MODEL)
        )
        perception_repository = PerceptionRepository(services.db_pool)
        experience_repository = ExperienceRepository(
            services.db_pool,
            consolidation_job_repository=(
                memory_job_repository if consolidation_enabled else None
            ),
        )
        memory_repository = None
        if legacy_memory:
            memory_repository = MemoryRepository(services.db_pool)
            services.memory_forgetting_service = MemoryForgettingService(
                memory_repository
            )
            memory_projection_repository = MemoryProjectionRepository(services.db_pool)
            services.memory_projection_service = MemoryProjectionService(
                repository=memory_projection_repository,
                limits=MemoryProjectionLimits(
                    max_characters=settings.MEMORY_PROFILE_MAX_CHARACTERS,
                    section_item_limit=settings.MEMORY_PROFILE_SECTION_ITEM_LIMIT,
                    recent_episode_limit=settings.MEMORY_PROFILE_RECENT_EPISODE_LIMIT,
                    superseded_history_limit=(
                        settings.MEMORY_PROFILE_SUPERSEDED_HISTORY_LIMIT
                    ),
                ),
            )
            if (
                memory_job_registry is not None
                and memory_job_reconciler is not None
            ):
                memory_job_registry.register(
                    RefreshMemoryProfileJobHandler(
                        services.memory_projection_service
                    )
                )
                memory_job_reconciler.register(
                    MemoryProfileRecoveryRule(
                        projection_repository=memory_projection_repository,
                        batch_size=settings.MEMORY_PROFILE_RECOVERY_BATCH_SIZE,
                    )
                )
            episodic_repository = EpisodicMemoryRepository(
                services.db_pool,
                experience_repository=experience_repository,
                memory_repository=memory_repository,
            )
        embedding_model = None
        embedding_job_repository = None
        services.working_memory_service = WorkingMemoryService(
            repository=perception_repository,
            policy=WorkingMemoryPolicy(
                ttl_seconds=settings.WORKING_MEMORY_TTL_SECONDS,
                entity_ttl_seconds=settings.WORKING_MEMORY_ENTITY_TTL_SECONDS,
                visible_text_ttl_seconds=settings.WORKING_MEMORY_VISIBLE_TEXT_TTL_SECONDS,
                max_entities=settings.WORKING_MEMORY_MAX_ENTITIES,
                max_recent_changes=settings.WORKING_MEMORY_MAX_RECENT_CHANGES,
            ),
            domain_event_bus=services.domain_event_bus,
        )
        services.prospective_repository = ProspectiveRepository(services.db_pool)
        services.prospective_context_builder = ProspectiveContextBuilder(
            focus_reader=services.prospective_repository,
            reminder_reader=ReminderRepository(services.db_pool),
            goal_reader=memory_repository,
            owner_working_memory_reader=services.working_memory_service,
            max_conditional_reminders=(
                settings.PROSPECTIVE_MAX_CONDITIONAL_REMINDERS
            ),
            max_goals=settings.PROSPECTIVE_MAX_GOALS,
            max_commitments=settings.PROSPECTIVE_MAX_COMMITMENTS,
        )
        services.prospective_service = ProspectiveMemoryService(
            repository=services.prospective_repository,
            realtime_session_manager=services.realtime_session_manager,
            settings=settings,
        )
        services.experience_event_service = ExperienceEventService(
            repository=experience_repository,
            segmenter=ExperienceEventSegmenter(
                EventSegmentationPolicy(
                    idle_close_seconds=settings.PERCEPTION_EVENT_IDLE_CLOSE_SECONDS,
                    max_event_seconds=settings.PERCEPTION_EVENT_MAX_SECONDS,
                )
            ),
            domain_event_bus=services.domain_event_bus,
            working_memory_service=services.working_memory_service,
            keyframe_selector=DeterministicKeyframeSelector(),
            keyframe_policy=KeyframeRetentionPolicy(
                final_keyframe_max_count=settings.EVENT_KEYFRAME_MAX_COUNT,
                consolidation_candidate_max_count=(
                    settings.EVENT_CONSOLIDATION_KEYFRAME_MAX_COUNT
                ),
                open_event_candidate_max_count=(
                    settings.EVENT_KEYFRAME_CANDIDATE_MAX_COUNT
                ),
                candidate_retention_seconds=settings.EVENT_CANDIDATE_RETENTION_SECONDS,
                ordinary_keyframe_retention_seconds=(
                    settings.EVENT_ORDINARY_KEYFRAME_RETENTION_SECONDS
                ),
            ),
        )
        services.experience_event_sweeper = ExperienceEventSweeper(
            services.experience_event_service
        )
        try:
            storage_config = get_supabase_storage_config(settings)
        except JarvisHTTPError:
            logger.info("visual frame storage not configured")
        else:
            services.visual_frame_storage = VisualFrameStorageClient(storage_config)
            services.jarvis_file_storage = JarvisFileStorageClient(storage_config)
            visual_frame_retention_service = VisualFrameRetentionService(
                repository=VisualFrameRepository(services.db_pool),
                storage=services.visual_frame_storage,
            )
            if settings.MEMORY_VISUAL_FRAME_RETENTION_ENABLED:
                services.visual_frame_retention_task = asyncio.create_task(
                    run_visual_frame_retention_loop(
                        visual_frame_retention_service,
                        interval_seconds=(
                            settings.MEMORY_VISUAL_FRAME_EXPIRY_INTERVAL_SECONDS
                        ),
                    )
                )
            services.visual_memory_services = create_visual_memory_services(
                settings=settings,
                pool=services.db_pool,
                realtime_session_manager=services.realtime_session_manager,
                model_limiter=services.model_limiter,
                sensory_frame_window=services.sensory_frame_window,
                domain_event_bus=services.domain_event_bus,
                working_memory_service=services.working_memory_service,
                experience_event_service=services.experience_event_service,
                prospective_context_builder=services.prospective_context_builder,
                prospective_service=services.prospective_service,
            )
            logger.info("visual frame storage initialized")
        visual_services = services.visual_memory_services
        historical_model = (
            GeminiFastVisualModel(visual_services.gemini_client)
            if isinstance(visual_services, VisualMemoryServices)
            and isinstance(
                visual_services.gemini_client,
                GeminiVisualAnalyzerClient,
            )
            else None
        )
        historical_evidence_repository = HistoricalEvidenceRepository(services.db_pool)
        services.historical_image_service = HistoricalImageService(
            repository=historical_evidence_repository,
            image_loader=services.visual_frame_storage,
            max_image_bytes=settings.VISUAL_FRAME_MAX_BYTES,
            max_width=settings.VISUAL_FRAME_MAX_WIDTH,
            max_height=settings.VISUAL_FRAME_MAX_HEIGHT,
            request_timeout_seconds=settings.VISUAL_IMAGE_RETRIEVAL_TIMEOUT_SECONDS,
            max_frames=settings.MEMORY_REINSPECTION_MAX_KEYFRAMES,
        )
        # Preserve the explicit service API search(reinspect_evidence=True).
        # The speaking agent's search tool no longer exposes that route.
        services.historical_evidence_inspector = HistoricalEvidenceInspector(
            repository=historical_evidence_repository,
            image_loader=services.visual_frame_storage,
            model=historical_model,
            model_limiter=services.model_limiter,
            max_frames=settings.MEMORY_REINSPECTION_MAX_KEYFRAMES,
        )
        logger.info(
            "historical evidence reinspection initialized configured=%s",
            historical_model is not None and services.visual_frame_storage is not None,
        )
        if (
            consolidation_enabled
            and memory_job_registry is not None
            and memory_job_reconciler is not None
        ):
            assert memory_job_repository is not None
            structured_output_client = OpenAIStructuredOutputClient(
                api_key=settings.OPENAI_API_KEY or "",
                model=settings.OPENAI_CONSOLIDATION_MODEL or "",
                timeout_seconds=settings.OPENAI_CONSOLIDATION_TIMEOUT_SECONDS,
                limiter=services.model_limiter,
            )
            consolidation_service = ExperienceConsolidationService(
                experience_repository=experience_repository,
                perception_repository=perception_repository,
                client=structured_output_client,
                image_loader=services.visual_frame_storage,
                owner_working_memory_reader=services.working_memory_service,
                active_reminder_reader=ReminderRepository(services.db_pool),
                semantic_context_reader=memory_repository,
                max_keyframes=settings.MEMORY_CONSOLIDATION_MAX_KEYFRAMES,
                max_transcripts=settings.MEMORY_CONSOLIDATION_MAX_TRANSCRIPTS,
            )
            services.experience_consolidation_service = consolidation_service
            reflection_trigger_service = None
            try:
                embedding_config = gemini_multimodal_embedding_config_from_settings(
                    settings
                )
            except (GeminiMultimodalEmbeddingNotConfiguredError, ValueError) as error:
                logger.info("episodic memory embedding not configured: %s", error)
            else:
                embedding_model = GeminiMultimodalEmbeddingModel(embedding_config)
                embedding_service = ExperienceEventEmbeddingService(
                    memory_repository,
                    embedding_model,
                )
                services.experience_event_embedding_service = embedding_service
                semantic_repository = SemanticMemoryRepository(
                    services.db_pool,
                    supersede_min_confidence=(
                        settings.MEMORY_SEMANTIC_SUPERSEDE_MIN_CONFIDENCE
                    ),
                )
                semantic_reconciler = SemanticMemoryReconciler(
                    minimum_inferred_confidence=(
                        settings.MEMORY_SEMANTIC_INFERENCE_MIN_CONFIDENCE
                    )
                )
                services.semantic_memory_reconciler = semantic_reconciler
                memory_job_registry.register(
                    ReconcileSemanticMemoryJobHandler(
                        repository=semantic_repository,
                        reconciler=semantic_reconciler,
                        job_repository=memory_job_repository,
                        embedding_model=embedding_model,
                    )
                )
                memory_job_registry.register(
                    EmbedMemoryEntityJobHandler(
                        repository=semantic_repository,
                        model=embedding_model,
                    )
                )
                memory_job_registry.register(
                    EmbedExperienceEventJobHandler(
                        repository=episodic_repository,
                        service=embedding_service,
                        image_loader=services.visual_frame_storage,
                        job_repository=memory_job_repository,
                    )
                )
                memory_job_registry.register(
                    EmbedMemoryItemJobHandler(
                        repository=episodic_repository,
                        service=embedding_service,
                        job_repository=memory_job_repository,
                    )
                )
                memory_job_reconciler.register(SemanticMemoryRecoveryRule())
                embedding_job_repository = memory_job_repository
                reflection_repository = ReflectionRepository(services.db_pool)
                reflection_trigger_service = ReflectionTriggerService(
                    pool=services.db_pool,
                    repository=reflection_repository,
                    job_repository=memory_job_repository,
                    minimum_episodes=settings.MEMORY_REFLECTION_MIN_NEW_EPISODES,
                    maximum_episodes=settings.MEMORY_REFLECTION_MAX_EPISODES,
                )
                services.reflection_trigger_service = reflection_trigger_service
                memory_job_registry.register(
                    GenerateReflectionJobHandler(
                        repository=reflection_repository,
                        service=ReflectionGenerationService(
                            structured_output_client
                        ),
                        preference_repository=PreferenceRepository(
                            services.db_pool
                        ),
                        job_repository=memory_job_repository,
                        minimum_preference_episodes=(
                            settings.MEMORY_INFERRED_PREFERENCE_MIN_SUPPORTING_EPISODES
                        ),
                        minimum_preference_confidence=(
                            settings.MEMORY_INFERRED_PREFERENCE_MIN_CONFIDENCE
                        ),
                        minimum_procedure_episodes=(
                            settings.MEMORY_PROCEDURE_MIN_SUPPORTING_EPISODES
                        ),
                    )
                )
                memory_job_reconciler.register(
                    ReflectionRecoveryRule(reflection_trigger_service)
                )
                logger.info("episodic embedding and semantic memory reconciliation initialized")
            memory_job_registry.register(
                PromoteExperienceEventJobHandler(
                    repository=episodic_repository,
                    service=EpisodePromotionService(
                        minimum_importance=(
                            settings.MEMORY_EPISODE_PROMOTION_MIN_IMPORTANCE
                        ),
                        important_keyframe_minimum_importance=(
                            settings.MEMORY_IMPORTANT_KEYFRAME_MIN_IMPORTANCE
                        ),
                        final_keyframe_limit=settings.EVENT_KEYFRAME_MAX_COUNT,
                    ),
                    embedding_job_repository=embedding_job_repository,
                    reflection_trigger_service=reflection_trigger_service,
                    ordinary_keyframe_retention_seconds=(
                        settings.EVENT_ORDINARY_KEYFRAME_RETENTION_SECONDS
                    ),
                    skipped_event_retention_days=(
                        settings.MEMORY_SKIPPED_EVENT_RETENTION_DAYS
                    ),
                )
            )
            logger.info("episodic memory promotion initialized")
            memory_job_registry.register(
                ConsolidateExperienceEventJobHandler(
                    repository=experience_repository,
                    service=consolidation_service,
                    candidate_lease_seconds=settings.EVENT_CANDIDATE_RETENTION_SECONDS,
                    promotion_job_repository=memory_job_repository,
                )
            )
            memory_job_reconciler.register(
                ExperienceConsolidationRecoveryRule(
                    candidate_lease_seconds=settings.EVENT_CANDIDATE_RETENTION_SECONDS,
                )
            )
            logger.info("GPT experience consolidation initialized")
        elif settings.MEMORY_JOB_RUNNER_ENABLED and legacy_memory:
            logger.info("GPT experience consolidation not configured")
        if memory_job_repository is not None and memory_job_registry is not None:
            retention_repository = MemoryRetentionRepository()
            retention_policy = MemoryRetentionPolicy(
                observation_detail_days=(
                    settings.MEMORY_OBSERVATION_DETAIL_RETENTION_DAYS
                ),
                unreferenced_observation_days=(
                    settings.MEMORY_UNREFERENCED_OBSERVATION_RETENTION_DAYS
                ),
                unreferenced_visual_frame_days=(
                    settings.MEMORY_UNREFERENCED_VISUAL_FRAME_RETENTION_DAYS
                ),
                skipped_event_days=settings.MEMORY_SKIPPED_EVENT_RETENTION_DAYS,
                completed_job_days=settings.MEMORY_COMPLETED_JOB_RETENTION_DAYS,
                retrieval_audit_days=settings.MEMORY_RETRIEVAL_AUDIT_RETENTION_DAYS,
                batch_size=settings.MEMORY_MAINTENANCE_BATCH_SIZE,
            )
            for job_type in DATABASE_MAINTENANCE_JOB_TYPES:
                memory_job_registry.register(
                    MemoryMaintenanceJobHandler(
                        job_type=job_type,
                        repository=retention_repository,
                        job_repository=memory_job_repository,
                        policy=retention_policy,
                    )
                )
                maintenance_job_intervals[job_type] = (
                    settings.MEMORY_MAINTENANCE_INTERVAL_SECONDS
                )
        if legacy_memory:
            services.memory_retrieval_service = HierarchicalMemoryRetrievalService(
                memory_repository=memory_repository,
                perception_repository=perception_repository,
                candidate_collector=RetrievalCandidateCollector(
                    memory_repository=memory_repository,
                    perception_repository=perception_repository,
                    embedding_model=embedding_model,
                ),
                candidate_limit=settings.MEMORY_RETRIEVAL_CANDIDATE_LIMIT,
                result_limit=settings.MEMORY_RETRIEVAL_RESULT_LIMIT,
                default_seconds=settings.MEMORY_RETRIEVAL_DEFAULT_SECONDS,
                max_days=settings.MEMORY_RETRIEVAL_MAX_DAYS,
                evidence_inspector=services.historical_evidence_inspector,
                image_service=services.historical_image_service,
            )
            services.memory_service = MemoryApplicationService(
                repository=memory_repository,
                retrieval_service=services.memory_retrieval_service,
                session_repository=SessionRepository(services.db_pool),
                frame_window=services.sensory_frame_window,
                working_memory_service=services.working_memory_service,
                embedding_job_repository=embedding_job_repository,
                current_frame_max_age_seconds=(
                    settings.VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS
                ),
                entity_repository=EntityRepository(services.db_pool),
            )
        services.tool_modules = _create_tool_modules(services)
        services.assistant_context_providers = _create_assistant_context_providers(
            services
        )
        logger.info("tool modules initialized count=%s", len(services.tool_modules))
        if settings.EXPERIENCE_EVENT_SWEEPER_ENABLED:
            services.experience_event_sweeper_task = asyncio.create_task(
                run_experience_event_sweeper_loop(
                    services.experience_event_sweeper,
                    interval_seconds=settings.PERCEPTION_EVENT_SWEEP_INTERVAL_SECONDS,
                )
            )
        if (
            memory_job_repository is not None
            and memory_job_registry is not None
            and memory_job_reconciler is not None
        ):
            services.memory_job_runner = MemoryJobRunner(
                pool=services.db_pool,
                repository=memory_job_repository,
                registry=memory_job_registry,
                reconciler=memory_job_reconciler,
                config=MemoryJobRunnerConfig(
                    poll_interval_seconds=settings.MEMORY_JOB_POLL_INTERVAL_SECONDS,
                    max_concurrency=settings.MEMORY_JOB_MAX_CONCURRENCY,
                    claim_timeout_seconds=settings.MEMORY_JOB_CLAIM_TIMEOUT_SECONDS,
                    reconcile_interval_seconds=(
                        settings.MEMORY_JOB_RECONCILE_INTERVAL_SECONDS
                    ),
                ),
            )
            services.memory_job_runner_task = asyncio.create_task(
                services.memory_job_runner.run()
            )
            if settings.MEMORY_MAINTENANCE_ENABLED and maintenance_job_intervals:
                maintenance_scheduler = MemoryMaintenanceScheduler(
                    repository=memory_job_repository,
                    owner_id=settings.JARVIS_OWNER_ID,
                    job_intervals=maintenance_job_intervals,
                )
                services.memory_maintenance_scheduler_task = asyncio.create_task(
                    run_memory_maintenance_scheduler_loop(maintenance_scheduler)
                )
            logger.info(
                "memory job runner initialized handlers=%s",
                len(memory_job_registry.job_types),
            )
    else:
        logger.info("database pool not configured")
        services.assistant_context_providers = _create_assistant_context_providers(services)


async def _initialize_reminder_services(services: JarvisRuntimeServices) -> None:
    """Build generators only after final tool/context authority selection."""
    if services.db_pool is None:
        return
    settings = services.settings
    if (
        settings.REMINDER_SCHEDULER_ENABLED
        and is_configured_secret(settings.OPENAI_API_KEY)
        and is_configured_secret(settings.OPENAI_REMINDER_MODEL)
    ):
        reminder_repository = ReminderRepository(services.db_pool)
        visual_services = services.visual_memory_services
        reminder_delivery = ReminderDeliveryService(
            owner_id=settings.JARVIS_OWNER_ID,
            repository=reminder_repository,
            session_repository=SessionRepository(services.db_pool),
            realtime_session_manager=services.realtime_session_manager,
            generator=ReminderMessageGenerator(
                settings,
                tool_modules=services.tool_modules,
                context_providers=services.assistant_context_providers,
            ),
        )
        reminder_scheduler = ReminderScheduler(
            settings=settings,
            repository=reminder_repository,
            delivery=reminder_delivery,
        )
        services.reminder_delivery_service = reminder_delivery
        services.reminder_scheduler_task = asyncio.create_task(
            run_reminder_scheduler_loop(
                reminder_scheduler,
                interval_seconds=settings.REMINDER_SCHEDULER_INTERVAL_SECONDS,
            )
        )
    elif settings.REMINDER_SCHEDULER_ENABLED:
        logger.info("reminder scheduler not configured")


@asynccontextmanager
async def lifespan(
    app: FastAPI, *, retention_clock: Callable[[], datetime] | None = None,
) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.JARVIS_RESTORE_IN_PROGRESS:
        # Fail before database workers, models, HTTP/WebSocket admission or
        # source/context construction can expose a partially restored backup.
        raise RuntimeError("jarvis_restore_in_progress")
    logger.info("starting Jarvis backend environment=%s", settings.APP_ENV)
    sensory_frame_window = RecentVisualFrameWindow(
        max_frames_per_session=settings.VISUAL_SENSORY_WINDOW_MAX_FRAMES,
        ttl_seconds=settings.VISUAL_SENSORY_WINDOW_TTL_SECONDS,
    )
    realtime_session_manager = RealtimeSessionManager()
    domain_event_bus = DomainEventBus()
    domain_event_bus.subscribe(
        DomainEventName.VISUAL_FRAME_ACCEPTED,
        realtime_session_manager.handle_visual_frame_accepted,
    )
    domain_event_bus.subscribe(
        DomainEventName.WORKING_MEMORY_UPDATED,
        realtime_session_manager.handle_working_memory_updated,
    )
    domain_event_bus.subscribe(
        DomainEventName.OWNER_WORKING_MEMORY_UPDATED,
        realtime_session_manager.handle_owner_working_memory_updated,
    )
    domain_event_bus.subscribe(
        DomainEventName.MEMORY_ITEM_CHANGED,
        realtime_session_manager.handle_memory_item_changed,
    )
    runtime_services = JarvisRuntimeServices(
        settings=settings,
        realtime_session_manager=realtime_session_manager,
        sensory_frame_window=sensory_frame_window,
        current_image_service=CurrentImageService(
            frame_window=sensory_frame_window,
            max_age_seconds=settings.VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS,
            max_image_bytes=settings.VISUAL_FRAME_MAX_BYTES,
            max_width=settings.VISUAL_FRAME_MAX_WIDTH,
            max_height=settings.VISUAL_FRAME_MAX_HEIGHT,
        ),
        latest_visual_frame_buffer=LatestVisualFrameBuffer(
            ttl_seconds=settings.VISUAL_LATEST_FRAME_TTL_SECONDS,
            frame_window=sensory_frame_window,
        ),
        domain_event_bus=domain_event_bus,
        model_limiter=PriorityAsyncLimiter(
            settings.MODEL_PROVIDER_MAX_CONCURRENCY
            or settings.VISUAL_ANALYZER_MAX_CONCURRENCY
        ),
    )
    app.state.runtime_services = runtime_services
    from app.media_audio import SoloistMediaService
    app.state.media_audio = SoloistMediaService.configured(settings.JARVIS_OWNER_ID)
    hermes_runtime = getattr(app.state, "hermes_profile_runtime", None)
    hermes_stop, hermes_task = asyncio.Event(), None
    try:
        await _initialize_runtime_services(runtime_services)
        await initialize_task_runtime(runtime_services)
        if retention_clock is None:
            await initialize_knowledge_runtime(runtime_services)
        else:
            await initialize_knowledge_runtime(runtime_services, retention_clock=retention_clock)
        configure_selected_memory(runtime_services)
        initialize_plugin_runtime(runtime_services)
        await initialize_recording_runtime(runtime_services)
        from app.live_state.companion_runtime import (
            initialize_companion, connect_companion_tools, connect_companion_views)
        await initialize_companion(runtime_services)
        from app.shared_memory.configuration import initialize_shared_memory
        await initialize_shared_memory(runtime_services)
        if runtime_services.companion_state is not None:
            from app.perception.companion_runtime import initialize_companion_perception
            runtime_services.companion_perception = initialize_companion_perception(
                state=runtime_services.companion_state,
                visual_services=runtime_services.visual_memory_services,
                memory=runtime_services.shared_memory,
                intake=runtime_services.shared_memory_intake,
                on_reminder=runtime_services.companion_notices.on_reminder,
                model_limiter=runtime_services.model_limiter)
        from app.runtime.workspace_delivery import initialize_workspace_presentations
        initialize_workspace_presentations(runtime_services)
        if runtime_services.companion_state is not None:
            from app.app_controls import AppControlService
            runtime_services.app_controls = AppControlService(
                owner_id=settings.JARVIS_OWNER_ID, pool=runtime_services.db_pool,
                sandbox_views=None, presentations=runtime_services.workspace_presentations,
                media_audio=app.state.media_audio)
        connect_companion_tools(runtime_services)
        from app.runtime.sandbox_viewing import initialize_sandbox_views
        await initialize_attention_runtime(runtime_services)
        await _initialize_reminder_services(runtime_services)
        if hermes_runtime is None:
            hermes_runtime = configured_profile_runtime(settings=settings, runtime_services=runtime_services)
            app.state.hermes_profile_runtime = hermes_runtime
        runtime_services.hermes_profile_runtime = hermes_runtime
        initialize_sandbox_views(runtime_services, retained_runtime=hermes_runtime)
        connect_companion_views(runtime_services)
        if hermes_runtime is not None:
            # Share this server and its lifecycle; native readiness must not
            # delay unrelated phone/API startup.
            hermes_task = asyncio.create_task(hermes_runtime.run(hermes_stop), name="hermes-profile")
        yield
    finally:
        if hermes_task is not None:
            hermes_stop.set()
            try:
                await asyncio.wait_for(asyncio.shield(hermes_task), timeout=10)
            except (TimeoutError, asyncio.CancelledError):
                hermes_task.cancel()
                await asyncio.gather(hermes_task, return_exceptions=True)
            except Exception:
                logger.error("Hermes profile stopped with an error")
        if app.state.media_audio is not None:
            await app.state.media_audio.close()
        await runtime_services.close()
        logger.info("stopping Jarvis backend")


def create_app(*, retention_clock: Callable[[], datetime] | None = None,
               hermes_profile_runtime=None) -> FastAPI:
    """Bind a trusted retention-only dependency before startup, never from requests."""
    if retention_clock is not None and not callable(retention_clock):
        raise ValueError("Retention clock must be callable")
    if hermes_profile_runtime is not None and (
        not callable(getattr(hermes_profile_runtime, "run", None))
        or not callable(getattr(hermes_profile_runtime, "current_events", None))
    ):
        raise TypeError("Hermes runtime must be supplied by trusted server composition")
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL, json_logs=settings.LOG_JSON)

    app = FastAPI(
        title=settings.APP_NAME,
        lifespan=(lifespan if retention_clock is None else
                  partial(lifespan, retention_clock=retention_clock)),
    )
    app.state.hermes_profile_runtime = hermes_profile_runtime

    if settings.CORS_ALLOW_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ALLOW_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def log_http_request(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id") or str(uuid4())
        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "HTTP request failed",
                extra={
                    "event": "http_request_failed",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(
                        (time.perf_counter() - started_at) * 1_000,
                        2,
                    ),
                },
            )
            raise
        response.headers["X-Request-ID"] = request_id
        log_method = logger.debug if request.url.path == "/health" else logger.info
        log_method(
            "HTTP request completed",
            extra={
                "event": "http_request_completed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started_at) * 1_000, 2),
            },
        )
        return response

    @app.exception_handler(JarvisHTTPError)
    async def handle_jarvis_http_error(
        request: Request,
        exc: JarvisHTTPError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    app.include_router(health_router)
    app.include_router(background_wake_router(lambda: (
        app.state.hermes_profile_runtime.current_events()
        if app.state.hermes_profile_runtime is not None else None)))
    app.include_router(behavior_profile_router)
    app.include_router(plugins_router)
    app.include_router(diagnostics_router)
    app.include_router(sessions_router)
    app.include_router(reminders_router)
    from app.live_state.location_router import router as live_state_location_router
    app.include_router(live_state_location_router)
    from app.app_controls import router as app_controls_router
    app.include_router(app_controls_router)
    from app.media_audio import router as media_audio_router
    app.include_router(media_audio_router)
    app.include_router(memory_router)
    app.include_router(files_router)
    app.include_router(recording_router)
    app.include_router(google_calendar_router)
    app.include_router(realtime_router)
    app.include_router(visual_memory_router)
    app.include_router(tasks_router)
    app.include_router(attention_router)
    app.include_router(projects_router)
    app.include_router(workspace_router)
    app.include_router(sandbox_view_router)
    from app.auth.client_context import router as client_context_router
    app.include_router(client_context_router)
    return app


from app.hermes_sessions.serving import configured_profile_runtime

app = create_app()
