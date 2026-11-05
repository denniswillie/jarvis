"""Read-only account catalog using the process's existing context authority.

Default construction performs no I/O and creates no session, worker or user policy.
Explicit managed-media selection reads only bounded operator-owned local evidence.
Managed-use enforcement must be integrated before enabling policy mutations.
"""
from app.plugins.authority import PluginAuthority
from app.plugins.deployment import CURATED_DEPLOYMENT_BASELINE_V1
from app.plugins.repository import PluginPolicyRepository
from app.plugins.service import PluginPolicyService
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from app.tasks.models import TaskError
from app.plugins.media_deployment import load_managed_media


def initialize_plugin_runtime(services):
    services.plugin_policy_service = None
    services.plugin_authority = None
    task_service = getattr(services, "background_task_service", None)
    if task_service is not None:
        task_service.managed_plugins = None
    pool = services.db_pool
    context_service = services.background_task_context_service
    contexts = getattr(context_service, "repository", None)
    if contexts is None:
        contexts = getattr(services.knowledge_runtime, "context_repository", None)
    if pool is None or not isinstance(contexts, ContextRepository):
        return
    tasks = contexts.tasks
    if not isinstance(tasks, TaskRepository) or tasks.pool is not pool:
        return

    repository = PluginPolicyRepository(pool)
    authority = PluginAuthority(repository, contexts, tasks,
        deployment_baseline=CURATED_DEPLOYMENT_BASELINE_V1)
    services.plugin_authority = authority
    services.plugin_policy_service = PluginPolicyService(
        repository, authority=authority, changes_enabled=False)
    if task_service is not None:
        # The ordinary API TaskService already exists. Media is an optional
        # binding; its configuration must not take conversation/notes offline.
        try:
            task_service.managed_plugins = load_managed_media(authority,
                owner_id=services.settings.JARVIS_OWNER_ID,
                sandbox_configuration=task_service.sandbox_configuration,
                artifact_configuration=task_service.artifact_configuration)
        except TaskError:
            pass
