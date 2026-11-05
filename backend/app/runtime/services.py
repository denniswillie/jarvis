from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable

import asyncpg

from app.files.storage import JarvisFileStorageClient
from app.assistant.context_providers import AssistantContextProvider
from app.realtime.session_manager import RealtimeSessionManager
from app.reminders.delivery import ReminderDeliveryService
from app.perception.frame_window import RecentVisualFrameWindow
from app.runtime.concurrency import PriorityAsyncLimiter
from app.runtime.feature_config import MemoryAuthority
from app.runtime.events import DomainEventBus
from app.settings import JarvisSettings
from app.tools.module import ToolModule
from app.visual_memory.latest_frame_buffer import LatestVisualFrameBuffer
from app.visual_memory.services import VisualMemoryServices
from app.visual_memory.storage import VisualFrameStorageClient


LOGGER = logging.getLogger(__name__)


if TYPE_CHECKING:
    from app.experience.service import ExperienceEventService
    from app.experience.sweeper import ExperienceEventSweeper
    from app.memory.embeddings import ExperienceEventEmbeddingService
    from app.memory.consolidation.service import ExperienceConsolidationService
    from app.memory.jobs.runner import MemoryJobRunner
    from app.memory.projection import MemoryProjectionService
    from app.memory.retrieval.service import HierarchicalMemoryRetrievalService
    from app.memory.service import MemoryApplicationService
    from app.memory.forgetting import MemoryForgettingService
    from app.memory.retrieval.evidence_inspector import HistoricalEvidenceInspector
    from app.memory.retrieval.image_evidence import HistoricalImageService
    from app.memory.reflection.service import ReflectionTriggerService
    from app.memory.prospective.context import ProspectiveContextBuilder
    from app.memory.prospective.repository import ProspectiveRepository
    from app.memory.prospective.service import ProspectiveMemoryService
    from app.memory.semantic.reconciler import SemanticMemoryReconciler
    from app.perception.current_image import CurrentImageService
    from app.perception.services import WorkingMemoryService
    from app.tasks.catalog import ConfiguredTaskProvider
    from app.tasks.service import TaskService
    from app.tasks.session_context import SessionTaskContextService
    from app.knowledge.runtime import KnowledgeRuntime
    from app.attention.service import AttentionService
    from app.workspace.presentation import PresentationService
    from app.plugins.authority import PluginAuthority
    from app.plugins.service import PluginPolicyService


@dataclass
class JarvisRuntimeServices:
    """Process-scoped resources constructed and owned by FastAPI lifespan."""

    settings: JarvisSettings
    realtime_session_manager: RealtimeSessionManager
    sensory_frame_window: RecentVisualFrameWindow
    latest_visual_frame_buffer: LatestVisualFrameBuffer
    domain_event_bus: DomainEventBus
    model_limiter: PriorityAsyncLimiter
    memory_authority: MemoryAuthority = MemoryAuthority.LEGACY
    db_pool: asyncpg.Pool | None = None
    behavior_profile_service: object | None = None
    behavior_profile_updates: object | None = None
    plugin_policy_service: PluginPolicyService | None = None
    plugin_authority: PluginAuthority | None = None
    visual_frame_storage: VisualFrameStorageClient | None = None
    jarvis_file_storage: JarvisFileStorageClient | None = None
    visual_memory_services: VisualMemoryServices | None = None
    current_image_service: CurrentImageService | None = None
    working_memory_service: WorkingMemoryService | None = None
    experience_event_service: ExperienceEventService | None = None
    experience_event_embedding_service: ExperienceEventEmbeddingService | None = None
    experience_consolidation_service: ExperienceConsolidationService | None = None
    semantic_memory_reconciler: SemanticMemoryReconciler | None = None
    reflection_trigger_service: ReflectionTriggerService | None = None
    historical_evidence_inspector: HistoricalEvidenceInspector | None = None
    historical_image_service: HistoricalImageService | None = None
    prospective_repository: ProspectiveRepository | None = None
    prospective_context_builder: ProspectiveContextBuilder | None = None
    prospective_service: ProspectiveMemoryService | None = None
    experience_event_sweeper: ExperienceEventSweeper | None = None
    experience_event_sweeper_task: asyncio.Task[None] | None = None
    memory_job_runner: MemoryJobRunner | None = None
    memory_job_runner_task: asyncio.Task[None] | None = None
    memory_maintenance_scheduler_task: asyncio.Task[None] | None = None
    memory_retrieval_service: HierarchicalMemoryRetrievalService | None = None
    memory_service: MemoryApplicationService | None = None
    memory_forgetting_service: MemoryForgettingService | None = None
    memory_projection_service: MemoryProjectionService | None = None
    tool_modules: tuple[ToolModule, ...] = ()
    assistant_context_providers: tuple[AssistantContextProvider, ...] = ()
    reminder_delivery_service: ReminderDeliveryService | None = None
    visual_frame_retention_task: asyncio.Task[None] | None = None
    reminder_scheduler_task: asyncio.Task[None] | None = None
    background_task_service: TaskService | None = None
    background_task_context_service: SessionTaskContextService | None = None
    background_task_providers: dict[str, ConfiguredTaskProvider] = field(default_factory=dict)
    background_task_provider_issue: str | None = None
    background_task_issue: str | None = None
    knowledge_runtime: KnowledgeRuntime | None = None
    knowledge_issue: str | None = None
    recording_admission: object | None = None
    recording_ingress: object | None = None
    recording_policy: object | None = None
    recording_issue: str | None = None
    recording_source_access: object | None = None
    attention_service: AttentionService | None = None
    attention_issue: str | None = None
    workspace_presentations: PresentationService | None = None
    sandbox_views: object | None = None
    live_work_runtime: object | None = None
    hermes_profile_runtime: object | None = None
    live_target_directory: dict | None = None
    companion_state: object | None = None
    companion_plugin_service: object | None = None
    companion_tools: object | None = None
    companion_notices: object | None = None
    companion_perception: object | None = None
    app_controls: object | None = None
    shared_memory: object | None = None
    shared_memory_intake: object | None = None
    shared_memory_task: asyncio.Task[None] | None = None
    shared_memory_stop: asyncio.Event | None = None
    shared_memory_invalidator: Callable[..., Awaitable[None]] | None = None
    shared_memory_issue: str | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    async def close(self) -> None:
        """Close constructed resources once, in reverse construction order."""

        if self._closed:
            return
        self._closed = True

        if self.behavior_profile_updates is not None:
            await self._close_safely("behavior profile updates", self.behavior_profile_updates.close)

        if self.attention_service is not None:
            await self._close_safely("attention delivery", self.attention_service.close)

        # Gemini owns admitted domain work independently of its media transport.
        # Its opt-in drain must run before experience, storage, and pool teardown.
        # GPT has no such hook and retains the existing close_all ordering below.
        prepare = getattr(self.realtime_session_manager, "prepare_for_resource_close", None)
        if prepare is not None:
            await self._close_safely("realtime domain work", prepare)

        close_context_sessions = (
            self.background_task_context_service is not None or self.knowledge_runtime is not None
            or self.attention_service is not None or self.recording_source_access is not None
            or self.shared_memory_intake is not None
        )
        if close_context_sessions:
            # Canonical capture must drain while its database is still alive,
            # including the S03-only path without a memory provider.
            await self._close_safely("realtime session manager", self.realtime_session_manager.close_all)
        await self._cancel_task(
            "memory maintenance scheduler",
            self.memory_maintenance_scheduler_task,
        )
        await self._cancel_task("memory job runner", self.memory_job_runner_task)
        if self.experience_consolidation_service is not None:
            await self._close_safely(
                "experience consolidation service",
                self.experience_consolidation_service.close,
            )
        await self._cancel_task(
            "experience event sweeper",
            self.experience_event_sweeper_task,
        )
        await self._cancel_task("reminder scheduler", self.reminder_scheduler_task)
        if self.reminder_delivery_service is not None:
            await self._close_safely(
                "reminder delivery",
                self.reminder_delivery_service.close,
            )
        await self._cancel_task("visual frame retention", self.visual_frame_retention_task)
        if self.visual_memory_services is not None:
            await self._close_safely(
                "visual memory services",
                self.visual_memory_services.close,
            )
        if self.experience_event_embedding_service is not None:
            await self._close_safely(
                "experience event embedding service",
                self.experience_event_embedding_service.close,
            )
        if self.experience_event_service is not None:
            await self._close_safely(
                "experience event service",
                self.experience_event_service.close,
            )
        # Producers must finish their canonical writes before memory intake closes.
        if self.live_work_runtime is not None:
            await self._close_safely("Live work transport", self.live_work_runtime.close)
        if self.companion_plugin_service is not None:
            await self._close_safely("companion plugin service", self.companion_plugin_service.close)
        if self.sandbox_views is not None:
            await self._close_safely("sandbox views", self.sandbox_views.close)
        if self.recording_source_access is not None:
            await self._close_safely("recording source access", self.recording_source_access.close)
        if self.knowledge_runtime is not None:
            await self._close_safely("knowledge runtime", self.knowledge_runtime.close)
        if self.shared_memory is not None or self.shared_memory_intake is not None:
            from app.shared_memory.configuration import close_shared_memory
            await self._close_safely("native shared memory", lambda: close_shared_memory(self))
        if self.jarvis_file_storage is not None:
            await self._close_safely("Jarvis file storage", self.jarvis_file_storage.close)
        if self.visual_frame_storage is not None:
            await self._close_safely("visual frame storage", self.visual_frame_storage.close)
        if self.db_pool is not None:
            await self._close_safely("database pool", self.db_pool.close)
        if not close_context_sessions:
            await self._close_safely(
                "realtime session manager",
                self.realtime_session_manager.close_all,
            )

    @staticmethod
    async def _cancel_task(label: str, task: asyncio.Task[None] | None) -> None:
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return
        except Exception:
            LOGGER.exception("failed to stop %s", label)

    @staticmethod
    async def _close_safely(label: str, close: Callable[[], Awaitable[None]]) -> None:
        try:
            await close()
        except Exception:
            LOGGER.exception("failed to close %s", label)
