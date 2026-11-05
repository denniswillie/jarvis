"""Opt-in rule materialization; existing reminder/perception services own delivery."""
import asyncio
import logging

from app.reminders.duties import ResponsibilityRepository
from app.reminders.duty_tools import ResponsibilityToolModule
from app.reminders.episode_tools import EpisodeCheckToolModule
from app.runtime.feature_config import Capability

LOGGER = logging.getLogger(__name__)


class ResponsibilityRuntime:
    def __init__(self, repository, owner, *, episode_checks_enabled=False):
        self.repository = repository
        self.owner = owner
        self.episode_checks = repository.episode_checks(enabled=episode_checks_enabled)
        self.issue = None
        self.task = None

    def start(self):
        self.task = asyncio.create_task(self.run())

    async def run(self):
        while True:
            try:
                await self.repository.materialize(self.owner)
                self.issue = None
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.issue = "responsibility_materialization_failed"
                LOGGER.warning("Responsibility scheduler failed error_type=%s", type(error).__name__)
            await asyncio.sleep(1)

    async def close(self):
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)


async def initialize_responsibilities(services, flags):
    resolution = flags.resolve(Capability.RESPONSIBILITIES, database_url=services.settings.SUPABASE_DB_URL)
    services.responsibility_issue = resolution.issue or "responsibilities_disabled"
    if not resolution.enabled or not resolution.configured or services.db_pool is None:
        return None
    episode_enabled = bool(getattr(services.settings, "RESPONSIVE_VOICE_ENABLED", False))
    runtime = ResponsibilityRuntime(ResponsibilityRepository(services.db_pool),
        services.settings.JARVIS_OWNER_ID, episode_checks_enabled=episode_enabled)
    try:
        await runtime.repository.materialize(runtime.owner)
        modules = [ResponsibilityToolModule(runtime.repository)]
        if episode_enabled:
            # A failed schema probe does not disable existing scheduled duties.
            # The episode owner, not the feature flag, reports actual capability.
            await runtime.episode_checks.initialize()
            modules.append(EpisodeCheckToolModule(runtime.episode_checks))
        services.tool_modules = (*getattr(services, "tool_modules", ()), *modules)
        runtime.start()
        services.responsibility_issue = None
        return runtime
    except Exception as error:
        services.responsibility_issue = "responsibilities_initialization_failed"
        LOGGER.warning("Responsibility initialization failed error_type=%s", type(error).__name__)
        await runtime.close()
        return None
