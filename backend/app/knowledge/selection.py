"""Explicit foreground memory authority, independent of shadow ingestion."""
from dataclasses import dataclass, replace
import json
from typing import ClassVar

from app.assistant.context_providers import ContextSection
from app.knowledge.area_tools import KnowledgeAreaToolModule
from app.projects.tools import ProjectToolModule
from app.knowledge.conversation_writes import ConversationMemoryWriteBackend
from app.knowledge.lifecycle_tools import SourceLifecycleToolModule
from app.knowledge.retention_tools import SourceRetentionToolModule
from app.knowledge.models import SourceError
from app.knowledge.measurements import MeasurementRepository, MeasurementToolModule
from app.knowledge.observations import ObservationPublication, UnavailableObservationPublication
from app.files.tools import FileToolModule
from app.knowledge.files import SourceFileRepository
from app.perception.current_image import CurrentImageService
from app.visual_memory.tools import VisualCaptureToolModule
from app.knowledge.images import SourceImageService
from app.knowledge.image_tools import SourceImageToolModule
from app.tasks.stored_images import StoredFrameImageService
from app.tasks.models import TaskError
from app.memory.tools import MemoryToolModule
from app.runtime.feature_config import MemoryAuthority


class UnavailableSelectedMemory:
    """An unavailable selected backend is never an empty result or legacy fallback."""
    async def search(self, **kwargs):
        self._unavailable()

    async def explain(self, **kwargs):
        self._unavailable()

    async def remember(self, **kwargs):
        self._unavailable()

    @staticmethod
    def _unavailable():
        raise SourceError("selected_memory_unavailable",
            "The selected source-backed memory is unavailable. Nothing was saved or searched; do not use legacy memory.", 503)


class UnavailableSelectedFiles(UnavailableSelectedMemory):
    list_limit = 8

    def __init__(self, legacy_repository):
        self.legacy_repository = legacy_repository

    async def create_text_file(self, **kwargs):
        self._unavailable()

    async def capture_current_frame(self, **kwargs):
        self._unavailable()

    async def list_files(self, **kwargs):
        self._unavailable()

    async def get_file(self, **kwargs):
        self._unavailable()

    async def delete_source_file(self, **kwargs):
        self._unavailable()


@dataclass(frozen=True)
class MemoryAuthorityContextProvider:
    services: object
    key: ClassVar[str] = "memory_authority"
    priority: ClassVar[int] = 111
    critical: ClassVar[bool] = False

    async def build_section(self, request):
        runtime = self.services.knowledge_runtime
        payload = {
            "authority": "hindsight", "legacy_fallback": False,
            "configured": runtime is not None,
            "issue": self.services.knowledge_issue,
            "meaning": (
                "Use source-backed search_memory, explain_memory and explicit remember. Configuration does not "
                "prove provider readiness. Saves acknowledge local storage, not completed native ingestion. "
                "The old generated profile and goal projection are not memory authorities. Empty prospective "
                "goals are not an exhaustive list of the user's commitments; use source-backed lookup. "
                "Do not claim that saved text independently proves a physical action happened. "
                "For exact dated numerical records use save_measurement, list_measurements and calculate_measurements. "
                "Calculation results cover only their selected entries; do not sum an incomplete history page. "
                "Knowledge-area tools manage area descriptions and visibility; creating or renaming an area "
                "does not switch the conversation's source scope."
            ),
        }
        if getattr(self.services, "background_task_service", None) is not None:
            payload["correction_route"] = (
                "For corrections to existing memories or exact measurements, use start_background_task with the memory_edit capability. "
                "save_measurement only records a new entry; it must not duplicate a correction. "
                "Pass the user's requested change and relevant conversation, not a full replacement record. "
                "The frontier worker reads the originals, preserves unrelated facts and authors the result for authorized delivery by the selected voice route. "
                "You may first ask a simple question to clarify which subject or version the user means. "
                "If an existing task asks a question, pass the user's answer through update_background_task. "
                "Do not substitute remember or a new event for correcting an existing source. "
                "If delegation is unavailable, explain that the correction has not been made; do not bypass it."
            )
        return ContextSection(self.key, payload, self.priority, request.requested_at_utc,
                              len(json.dumps(payload, ensure_ascii=True)))

    def fallback_section(self, request):
        payload = {"authority": "hindsight", "configured": False, "legacy_fallback": False,
                   "meaning": "Selected memory status is unknown, not empty. Do not fall back to old memories."}
        return ContextSection(self.key, payload, self.priority, request.requested_at_utc,
                              len(json.dumps(payload, ensure_ascii=True)))


def configure_selected_memory(services):
    visual = getattr(services, "visual_memory_services", None)
    task_service = getattr(services, "background_task_service", None)
    if services.memory_authority == MemoryAuthority.LEGACY:
        if task_service is not None:
            task_service.knowledge_area_resolver = None
        if visual is not None:
            visual.pipeline.set_observation_publication(None)
        # Shadow ingestion must not advertise a second conversational authority.
        modules = getattr(services, "tool_modules", ())
        providers = getattr(services, "assistant_context_providers", ())
        if any(module.name in {SourceLifecycleToolModule.name, SourceImageToolModule.name,
                               KnowledgeAreaToolModule.name, ProjectToolModule.name, SourceRetentionToolModule.name, MeasurementToolModule.name} for module in modules):
            services.tool_modules = tuple(module for module in modules
                if module.name not in {SourceLifecycleToolModule.name, SourceImageToolModule.name,
                                       KnowledgeAreaToolModule.name, ProjectToolModule.name, SourceRetentionToolModule.name, MeasurementToolModule.name})
        if any(provider.key in {"knowledge_sources", MemoryAuthorityContextProvider.key} for provider in providers):
            services.assistant_context_providers = tuple(provider for provider in providers
                if provider.key not in {"knowledge_sources", MemoryAuthorityContextProvider.key})
        def restore_file(module):
            if isinstance(module, FileToolModule) and isinstance(module.repository, (SourceFileRepository, UnavailableSelectedFiles)):
                return replace(module, repository=module.repository.legacy_repository)
            if isinstance(module, VisualCaptureToolModule) and isinstance(module.file_repository, (SourceFileRepository, UnavailableSelectedFiles)):
                return replace(module, file_repository=module.file_repository.legacy_repository)
            return module
        current_modules = getattr(services, "tool_modules", ())
        restored = tuple(restore_file(module) for module in current_modules)
        if any(before is not after for before, after in zip(current_modules, restored, strict=True)):
            services.tool_modules = restored
        return
    if services.memory_authority != MemoryAuthority.HINDSIGHT:
        raise ValueError("invalid_memory_authority")
    if any(getattr(services, name, None) is not None for name in (
        "memory_service", "memory_retrieval_service", "memory_projection_service", "memory_forgetting_service",
        "memory_job_runner", "memory_job_runner_task", "memory_maintenance_scheduler_task",
        "experience_consolidation_service", "semantic_memory_reconciler", "reflection_trigger_service",
    )):
        raise RuntimeError("legacy_memory_constructed_after_hindsight_selection")
    # Replacement rather than addition: the existing tool catalog enforces that
    # a selected backend cannot also receive any legacy repositories/services.
    modules = tuple(module for module in services.tool_modules
                    if module.name not in {MemoryToolModule.name, SourceImageToolModule.name,
                                           SourceLifecycleToolModule.name, KnowledgeAreaToolModule.name, ProjectToolModule.name, SourceRetentionToolModule.name, MeasurementToolModule.name})
    runtime = services.knowledge_runtime
    if task_service is not None:
        async def task_area(connection, owner_id, request, event, *, command_admission=None):
            if runtime is None:
                raise TaskError("task_knowledge_scope_unavailable",
                    "Selected knowledge scope is unavailable; no new task was accepted.", 503)
            try:
                runtime.context_for(owner_id)
                arguments = ({"command_admission": command_admission}
                             if command_admission is not None else {})
                return await runtime.area_selections.task_area_in_transaction(
                    connection, owner_id, request, event, **arguments)
            except SourceError as error:
                raise TaskError(error.code, error.message, error.status_code) from None
        task_service.knowledge_area_resolver = task_area
    if visual is not None:
        visual.pipeline.set_observation_publication(
            ObservationPublication(runtime.context_repository, owner_id=services.settings.JARVIS_OWNER_ID,
                                   access_check=runtime.context_for)
            if runtime is not None else UnavailableObservationPublication())
    if runtime is None:
        backend = UnavailableSelectedMemory()
        memory = MemoryToolModule(read_backend=backend, write_backend=backend)
    else:
        owner = services.settings.JARVIS_OWNER_ID
        requests = runtime.context_repository.tasks
        modules += (MeasurementToolModule(MeasurementRepository(runtime.sources, requests,
                        owner_id=owner, area_id=runtime.configuration.area_id,
                        selections=runtime.area_selections, access_check=runtime.context_for)),
                    runtime.area_tool_module_for(owner),
                    SourceRetentionToolModule(runtime.sources, requests, owner, access_check=runtime.context_for))
        if getattr(runtime, "projects", None) is not None:
            modules += (ProjectToolModule(runtime.projects, owner, runtime.configuration.area_id,
                selections=runtime.area_selections, access_check=runtime.context_for),)
        memory = MemoryToolModule(
            read_backend=runtime.memory_read_backend_for(owner),
            write_backend=ConversationMemoryWriteBackend(
                runtime.sources, requests, owner_id=owner, area_id=runtime.configuration.area_id,
                access_check=runtime.context_for, selections=runtime.area_selections,
            ),
        )
        storage = getattr(services, "visual_frame_storage", None)
        stored = None
        if storage is not None:
            settings = services.settings
            stored = StoredFrameImageService(runtime.sources.pool, storage,
                max_bytes=settings.VISUAL_FRAME_MAX_BYTES, max_width=settings.VISUAL_FRAME_MAX_WIDTH,
                max_height=settings.VISUAL_FRAME_MAX_HEIGHT,
                timeout_seconds=settings.VISUAL_IMAGE_RETRIEVAL_TIMEOUT_SECONDS)
        modules += (SourceImageToolModule(SourceImageService(runtime.sources, stored, owner_id=owner,
            area_id=runtime.configuration.area_id, access_check=runtime.context_for, selections=runtime.area_selections),
            owner, runtime.configuration.area_id),)
        if not any(module.name == SourceLifecycleToolModule.name for module in modules):
            modules += (SourceLifecycleToolModule(
                runtime.sources, requests, owner, runtime.configuration.area_id,
                access_check=runtime.context_for, selections=runtime.area_selections,
                allow_foreground_correction=task_service is None,
            ),)
    adapters = {}
    def selected_files(module):
        if isinstance(module, FileToolModule):
            repository, storage = module.repository, module.storage
        elif isinstance(module, VisualCaptureToolModule):
            repository, storage = module.file_repository, module.file_storage
            if repository is None or storage is None:
                return module
        else:
            return module
        legacy = getattr(repository, "legacy_repository", repository)
        if id(storage) not in adapters:
            if runtime is None:
                adapter = UnavailableSelectedFiles(legacy)
            else:
                settings = services.settings
                latest = getattr(services, "latest_visual_frame_buffer", None)
                current = (CurrentImageService(frame_window=latest.frame_window,
                    max_age_seconds=settings.VISUAL_LATEST_CAPTURE_MAX_AGE_SECONDS,
                    max_image_bytes=settings.VISUAL_FRAME_MAX_BYTES, max_width=settings.VISUAL_FRAME_MAX_WIDTH,
                    max_height=settings.VISUAL_FRAME_MAX_HEIGHT) if latest is not None else None)
                adapter = SourceFileRepository(runtime.sources, storage, owner_id=settings.JARVIS_OWNER_ID,
                    area_id=runtime.configuration.area_id, access_check=runtime.context_for, current_images=current,
                    selections=runtime.area_selections)
                register = getattr(runtime, "set_file_publications", None)
                if register is not None:
                    register(adapter.publications)
            adapter.legacy_repository = legacy
            adapters[id(storage)] = adapter
        adapter = adapters[id(storage)]
        return (replace(module, repository=adapter) if isinstance(module, FileToolModule)
                else replace(module, file_repository=adapter))
    services.tool_modules = (*(selected_files(module) for module in modules), memory)
    providers = tuple(provider for provider in services.assistant_context_providers
                      if provider.key != MemoryAuthorityContextProvider.key)
    services.assistant_context_providers = (*providers, MemoryAuthorityContextProvider(services))
