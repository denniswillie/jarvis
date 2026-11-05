"""Opt-in viewer composition; never starts workers or allocates provider resources."""
import os
from pathlib import Path

from app.runtime.feature_config import TaskLimits
from app.sandbox.models import E2BConfiguration
from app.sandbox.view_access import SandboxViewAccess
from app.sandbox.retained_view_access import RetainedSandboxViewAccess
from app.sandbox.view_provider import E2BDesktopViewer
from app.sandbox.view_secrets import SandboxViewSecrets
from app.sandbox.view_service import SandboxViewService
from app.tasks.repository import TaskRepository


def initialize_sandbox_views(services, *, retained_runtime=None, repository=None, provider=None, secrets=None):
    """Call after tasks and workspace presentations; close before the SQL pool.

    Worker and API must explicitly share SANDBOX_VIEW_SECRET_ROOT. This enables
    read-only viewing, not task admission, execution, cleanup or auto-resume.
    """
    services.sandbox_views = None
    secrets = secrets if secrets is not None else SandboxViewSecrets.configured()
    if secrets is None:
        return
    presentations = getattr(services, "workspace_presentations", None)
    retained_runtime = retained_runtime if retained_runtime is not None else getattr(services, "hermes_profile_runtime", None)
    if retained_runtime is not None:
        pool = getattr(services, "db_pool", None)
        if pool is None:
            return
        access = RetainedSandboxViewAccess(retained_runtime, pool=pool, owner_id=services.settings.JARVIS_OWNER_ID)
        if provider is None:
            provider = E2BDesktopViewer(retained_runtime.sandboxes.provider, secrets)
    else:
        if presentations is None:
            return
        tasks = getattr(services, "background_task_service", None)
        repository = repository if repository is not None else getattr(tasks, "repository", None)
        if repository is None:
            pool = getattr(services, "db_pool", None)
            if pool is None:
                return
            repository = TaskRepository(pool, TaskLimits())
        if provider is None:
            key_file = os.environ.get("JARVIS_FRONTIER_E2B_API_KEY_FILE")
            if not key_file:
                return
            from app.sandbox.e2b_provider import E2BProvider
            provider = E2BDesktopViewer(E2BProvider(E2BConfiguration(api_key=Path(key_file).read_text().strip())), secrets)
        access = SandboxViewAccess(repository, owner_id=services.settings.JARVIS_OWNER_ID)
    views = SandboxViewService(access, provider)
    services.sandbox_views = views
    if presentations is None:
        return
    previous = presentations.resolver

    async def resolve(ctx, kind, artifact_id, version_id):
        if kind == "sandbox":
            return await views.resolve_presentation(ctx, artifact_id, version_id)
        return await previous(ctx, kind, artifact_id, version_id)

    presentations.resolver = resolve
