"""Owned canonical recording/Files access, without a memory client or workers."""
from app.knowledge.files import SourceFileRepository
from app.knowledge.models import SourceError
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.knowledge.sources import SourceRepository
from app.runtime.feature_config import TaskLimits
from app.tasks.repository import TaskRepository


class RecordingSourceAccess:
    def __init__(self, *, owner_id, sources, tasks, area_id, storage, policy, inspector):
        self.owner_id, self.sources, self.tasks = owner_id, sources, tasks
        self.area_selections = SessionKnowledgeAreas(sources, tasks, area_id)
        self.closed = False
        self.files = SourceFileRepository(sources, storage, owner_id=owner_id, area_id=area_id,
            access_check=self.check, selections=self.area_selections,
            media_policy=policy, media_inspector=inspector)

    @classmethod
    async def create(cls, services, *, area_id, policy, inspector):
        sources = SourceRepository(services.db_pool)
        existing_tasks = getattr(services, "background_task_service", None)
        tasks = existing_tasks.repository if existing_tasks is not None else TaskRepository(services.db_pool, TaskLimits())
        if tasks.pool is not sources.pool:
            raise SourceError("recording_scope_configuration_invalid", "Recording scope must share its database.", 503)
        owner = services.settings.JARVIS_OWNER_ID
        async with sources.transaction(owner) as connection:
            await sources._area(connection, owner, area_id)
        return cls(owner_id=owner, sources=sources, tasks=tasks, area_id=area_id,
            storage=services.jarvis_file_storage, policy=policy, inspector=inspector)

    def check(self, owner_id):
        if owner_id != self.owner_id:
            raise SourceError("recording_owner_mismatch", "Recording access belongs to a different account.", 403)
        if self.closed:
            raise SourceError("recording_access_closed", "Recording access is shutting down.", 503)

    async def close(self):
        # Storage and pool remain owned by the enclosing application. Held
        # repositories retain this same fence and cannot outlive its account.
        self.closed = True
