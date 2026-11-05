"""Opt-in runtime ownership, not a replacement agent loop or memory engine."""
from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from app.knowledge.current_context import CurrentSourceContext, SourceContextProvider
from app.knowledge.deletion import SourceDeletionService
from app.knowledge.hindsight import HindsightClient
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.models import SourceError
from app.knowledge.memory_backend import SourceMemoryReadBackend
from app.knowledge.outbox import OutboxRepository
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.retrieval import SourceMemoryRetrieval
from app.knowledge.retention import SourceRetentionService
from app.knowledge.sources import SourceRepository
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.projects.service import ProjectService
from app.migrations.runner import read_schema_readiness
from app.runtime.feature_config import Capability, KnowledgeConfiguration, TaskLimits, get_second_brain_settings
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository


LOGGER = logging.getLogger(__name__)


class KnowledgeRuntime:
    def __init__(self, *, owner_id, sources, tasks, client, configuration: KnowledgeConfiguration,
                 retention_clock=None):
        self.owner_id, self.configuration = owner_id, configuration
        self.sources, self.client = sources, client
        self.projects = ProjectService(sources)
        self.deletions = SourceDeletionService(sources)
        self.retention = SourceRetentionService(sources, tasks, clock=retention_clock)
        self.area_selections = SessionKnowledgeAreas(sources, tasks, configuration.area_id)
        self.context_repository = ContextRepository(
            tasks, publication=CanonicalSourcePublication(sources, configuration.area_id, self.area_selections),
        )
        self.current_context = CurrentSourceContext(sources)
        self.retrieval = SourceMemoryRetrieval(sources, client)
        self.outbox = OutboxRepository(
            sources, lease_seconds=configuration.worker_lease_seconds,
            max_in_flight=configuration.native_inflight_limit,
        )
        self.worker = IngestionWorker(
            self.outbox, client, request_deadline=configuration.worker_deadline_seconds,
        )
        self.worker_id = "memory-" + uuid4().hex
        self.stop = asyncio.Event()
        self.task = None
        self.file_task = None
        self.retention_task = None
        self.retention_issue = None
        self.file_publications = None
        self.file_issue = None
        self.issue = None
        self._closed = False

    @classmethod
    async def create(cls, *, pool, owner_id, configuration, tasks=None, retention_clock=None):
        async with pool.acquire() as connection:
            schema = await connection.fetchval("select current_schema()")
            readiness = await read_schema_readiness(connection, schema=schema)
        if not readiness.ready:
            raise SourceError("knowledge_schema_not_ready", "Apply the reviewed migrations before enabling knowledge.", 503)
        sources = SourceRepository(pool)
        async with sources.transaction(owner_id) as connection:
            # Management and reconciliation must survive an archived-area restart;
            # ordinary reads/publication still require an active area.
            await sources._area(connection, owner_id, configuration.area_id, allow_archived=True)
        tasks = tasks if tasks is not None else TaskRepository(pool, TaskLimits())
        if tasks.pool is not pool:
            raise SourceError("knowledge_context_pool_mismatch", "Canonical context and sources must share a database.", 503)
        client = HindsightClient(
            configuration.base_url, configuration.api_key.get_secret_value(),
            timeout=configuration.http_timeout_seconds,
            **({"private_http_origin": configuration.private_http_origin} if configuration.private_http_origin else {}),
        )
        try:
            return cls(owner_id=owner_id, sources=sources, tasks=tasks, client=client,
                       configuration=configuration, retention_clock=retention_clock)
        except BaseException:
            await client.close()
            raise

    def start(self):
        if self._closed or self.task is not None:
            raise RuntimeError("Knowledge runtime cannot start twice or after close")
        self.task = asyncio.create_task(
            self.worker.run(self.owner_id, self.worker_id, self.stop, poll_seconds=self.configuration.poll_seconds),
            name=self.worker_id,
        )
        self.task.add_done_callback(self._worker_finished)
        self.file_task = asyncio.create_task(self._reconcile_files(), name=self.worker_id + "-files")
        self.retention_task = asyncio.create_task(self._expire_sources(), name=self.worker_id + "-retention")

    def set_file_publications(self, service):
        if (service.sources is not self.sources or service.owner_id != self.owner_id
                or service.area_id != self.configuration.area_id or self._closed):
            raise ValueError("File recovery must match the live knowledge owner and area")
        self.file_publications = service

    async def _reconcile_files(self):
        while not self.stop.is_set():
            try:
                if self.file_publications is not None:
                    outcomes = await self.file_publications.reconcile_pending(owner_id=self.owner_id, limit=16)
                    self.file_issue = "file_cleanup_unresolved" if any(
                        item["phase"] == "unresolved" for item in outcomes) else None
            except Exception as error:
                self.file_issue = "file_reconciliation_failed"
                LOGGER.warning("file reconciliation failed type=%s", type(error).__name__)
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=max(5, self.configuration.poll_seconds))
            except TimeoutError:
                pass

    async def _expire_sources(self):
        # No default source enrollment. Waiting first also keeps startup/close
        # independent from a retention pass or a provider's availability.
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=60)
                return
            except TimeoutError:
                pass
            try:
                outcomes = await self.retention.process_due(self.owner_id, limit=16)
                if outcomes:
                    self.retention_issue = "source_retention_needs_attention" if any(
                        item["status"] != "expired" for item in outcomes) else None
            except Exception as error:
                self.retention_issue = "source_retention_failed"
                LOGGER.warning("source retention failed type=%s", type(error).__name__)

    def _worker_finished(self, task):
        if self.stop.is_set():
            return
        self.issue = ("memory_worker_failed" if task.cancelled() or task.exception() is not None
                      else "memory_worker_stopped")
        LOGGER.warning("knowledge worker stopped code=%s", self.issue)

    def context_for(self, owner_id):
        if owner_id != self.owner_id:
            raise SourceError("knowledge_owner_mismatch", "Knowledge capture is outside this runtime owner scope.", 403)
        if self._closed:
            raise SourceError("knowledge_intake_closed", "Knowledge capture is shutting down.", 503)
        return self.context_repository

    def memory_read_backend_for(self, owner_id):
        """Explicit experiment seam. Live tool routing stays legacy until M04."""
        self.context_for(owner_id)
        return SourceMemoryReadBackend(
            self.sources, self.retrieval, owner_id=owner_id,
            area_id=self.configuration.area_id, access_check=self.context_for,
            selections=self.area_selections,
        )

    def lifecycle_tool_module_for(self, owner_id):
        """Explicit selected-backend seam; live legacy routing remains unchanged."""
        from app.knowledge.lifecycle_tools import SourceLifecycleToolModule
        self.context_for(owner_id)
        return SourceLifecycleToolModule(
            self.sources, self.context_repository.tasks, owner_id, self.configuration.area_id,
            access_check=self.context_for, selections=self.area_selections,
        )

    def area_tool_module_for(self, owner_id):
        """Explicit management composition; does not change live tool routing."""
        from app.knowledge.areas import KnowledgeAreaService
        from app.knowledge.area_tools import KnowledgeAreaToolModule
        self.context_for(owner_id)
        return KnowledgeAreaToolModule(
            KnowledgeAreaService(self.sources, self.context_repository.tasks), owner_id,
            access_check=self.context_for, selections=self.area_selections,
        )

    async def diagnostics(self):
        return {
            "configured": True,
            "worker_running": self.task is not None and not self.task.done(),
            "closed": self._closed, "issue": self.issue,
            "queue": await self.outbox.diagnostics(self.owner_id),
            "transport": self.client.diagnostics(),
            "source_retention": {"running": self.retention_task is not None and not self.retention_task.done(),
                                 "last_pass_issue": self.retention_issue, "default_expiry": None},
            "file_recovery": {"running": self.file_task is not None and not self.file_task.done(),
                              "configured": self.file_publications is not None, "issue": self.file_issue},
            "meaning": "Running is not native acceptance or proof that any source is searchable.",
        }

    async def close(self):
        if self._closed:
            return
        self._closed = True
        self.stop.set()
        workers = {task for task in (self.task, self.file_task, self.retention_task) if task is not None}
        try:
            if workers:
                # Observe the worker without forwarding its cancellation to
                # the cleanup caller. Cancellation of this caller still propagates
                # after the finally block releases the worker and HTTP client.
                await asyncio.wait(workers, timeout=self.configuration.shutdown_seconds)
        finally:
            for task in workers:
                if not task.done():
                    task.cancel()
            if workers:
                await asyncio.gather(*workers, return_exceptions=True)
            await self.client.close()


async def initialize_knowledge_runtime(services, *, retention_clock=None):
    """Initialize memory; the optional trusted clock affects retention only."""
    if services.knowledge_runtime is not None:
        return
    resolution = get_second_brain_settings().resolve(
        Capability.KNOWLEDGE, database_url=services.settings.SUPABASE_DB_URL,
    )
    services.knowledge_issue = resolution.issue or "knowledge_disabled"
    if not resolution.enabled or not resolution.configured:
        return
    if services.db_pool is None:
        services.knowledge_issue = "knowledge_database_unavailable"
        return
    if not isinstance(resolution.configuration, KnowledgeConfiguration):
        services.knowledge_issue = "knowledge_configuration_invalid"
        return
    runtime = None
    try:
        task_service = services.background_task_service
        runtime = await KnowledgeRuntime.create(
            pool=services.db_pool, owner_id=services.settings.JARVIS_OWNER_ID,
            configuration=resolution.configuration,
            tasks=task_service.repository if task_service is not None else None,
            retention_clock=retention_clock,
        )
        provider = SourceContextProvider(runtime.current_context, resolution.configuration.area_id,
                                         selections=runtime.area_selections)
        if any(item.key == provider.key for item in services.assistant_context_providers):
            raise SourceError("knowledge_context_provider_conflict", "A source-context provider is already registered.", 503)
        runtime.start()
        services.knowledge_runtime = runtime
        services.assistant_context_providers = (*services.assistant_context_providers, provider)
        services.knowledge_issue = None
    except BaseException as error:
        if runtime is not None:
            await runtime.close()
        if not isinstance(error, Exception):
            raise
        services.knowledge_issue = error.code if isinstance(error, SourceError) else "knowledge_initialization_failed"
