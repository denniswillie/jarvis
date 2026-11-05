"""Flag-gated composition using existing owner working memory and reminders."""
import os

from app.live_state.companion import CompanionState
from app.live_state.companion_tools import CompanionTools
from app.perception.repository import PerceptionRepository
from app.reminders.repository import ReminderRepository


async def initialize_companion(services, *, environ=None):
    env = os.environ if environ is None else environ
    if env.get("JARVIS_COMPANION_STATE_ENABLED", "").lower() not in {"true", "1"}:
        return
    if services.db_pool is None:
        raise ValueError("Companion state requires the existing database")
    working_memory = PerceptionRepository(services.db_pool)
    state = CompanionState(owner_id=services.settings.JARVIS_OWNER_ID,
        working_memory=working_memory,
        reminders=ReminderRepository(services.db_pool))
    await state.initialize()
    services.companion_state = state
    services.shared_memory_invalidator = state.invalidate_memory
    from app.live_state.companion_notices import CompanionNotices
    services.companion_notices = CompanionNotices(state, services.realtime_session_manager)
    from app.plugins.companion_marketplace import CompanionPluginService
    plugins = CompanionPluginService(
        owner_id=services.settings.JARVIS_OWNER_ID,
        working_memory=working_memory,
        runtime_services=services,
    )
    await plugins.initialize()
    services.companion_plugin_service = plugins
    state.plugin_projection = plugins.live_state_projection


def connect_companion_tools(services):
    state = getattr(services, "companion_state", None)
    if state is not None:
        services.companion_tools = CompanionTools(state,
            memory_intake=getattr(services, "shared_memory_intake", None),
            app_controls=getattr(services, "app_controls", None))


def connect_companion_views(services):
    state = getattr(services, "companion_state", None)
    views = getattr(services, "sandbox_views", None)
    if state is not None and views is not None and callable(getattr(views, "cached_inventory", None)):
        state.sandbox_projection = lambda owner: views.cached_inventory(owner, limit=8)
        state.sandbox_refresh = getattr(views, "refresh_inventory", None)
    controls = getattr(services, "app_controls", None)
    if state is not None and controls is not None:
        controls.views = views
        state.app_projection = controls.projection
    plugins = getattr(services, "companion_plugin_service", None)
    hermes_runtime = getattr(services, "hermes_profile_runtime", None)
    if plugins is not None and hermes_runtime is not None and getattr(hermes_runtime, "sandboxes", None) is not None:
        hermes_runtime.sandboxes.on_ready = plugins.reconcile_sandbox
