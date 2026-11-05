"""Optional retained-profile composition for the existing FastAPI process.

The operator provisions and starts the native gateway separately. This factory
does not start a model, allocate a sandbox, or create a second HTTP service.
"""
import json
import os
import asyncio
from pathlib import Path


def _connect_master_notices(runtime, runtime_services):
    manager = getattr(runtime_services, "realtime_session_manager", None)
    if manager is not None:
        controls = getattr(runtime_services, "app_controls", None)
        if controls is not None:
            controls.send_link_notice = manager.send_companion_notice
        from app.realtime.live_retained import RetainedMasterNotices
        runtime.master_notices = RetainedMasterNotices(runtime, manager)
        runtime.events.on_admitted = runtime.master_notices.admitted
        runtime.host.on_pump = runtime.master_notices.pump
    return runtime


def configured_profile_runtime(*, settings=None, runtime_services=None, read_live_state=None):
    selected = os.environ.get("JARVIS_HERMES_PROFILE_HOME")
    if not selected:
        return None

    from app.settings import JarvisSettings
    from app.sandbox.models import E2BConfiguration, SandboxPolicy
    from .runtime import RetainedProfileRuntime
    from .sandbox_provider import RetainedE2BProvider
    from app.sandbox.view_secrets import SandboxViewSecrets

    settings = settings if settings is not None else JarvisSettings()
    home = Path(selected).expanduser().resolve(strict=True)
    config = json.loads((home / "jarvis-adapter.json").read_text())
    if config.get("owner_id") != settings.JARVIS_OWNER_ID:
        raise ValueError("Retained profile belongs to another backend owner")
    if config.get("domain_tools") and runtime_services is None:
        raise ValueError("Domain-enabled profiles require initialized runtime services")
    key = (home / "jarvis" / "callback.key").read_bytes()
    if len(key) != 32:
        raise ValueError("Retained profile needs its persistent 32-byte callback key")
    key_file = os.environ.get("JARVIS_FRONTIER_E2B_API_KEY_FILE")
    if not key_file:
        raise ValueError("Retained profile needs the configured E2B credential file")
    credential = Path(key_file).read_text().strip()
    if not credential:
        raise ValueError("E2B credential file is empty")
    policy = SandboxPolicy.model_validate({"template": "desktop", "lifetime_seconds": 3600,
                                          **config.get("sandbox_policy", {})})
    provider = RetainedE2BProvider(E2BConfiguration(api_key=credential), view_secrets=SandboxViewSecrets.configured())
    options = dict(provider=provider, sandbox_policy=policy,
        callback_endpoint=settings.PUBLIC_BASE_URL.rstrip("/") + "/hermes/background-wake", signing_key=key)
    if not config.get("domain_tools"):
        return _connect_master_notices(RetainedProfileRuntime(home, **options), runtime_services)

    from app.files.tools import FileToolModule
    from app.knowledge.files import SourceFileRepository
    from app.runtime.feature_config import TaskLimits
    from app.tasks.repository import TaskRepository
    from .client import HermesGatewayClient
    from .domain_scope import DomainInputScope
    from .plugin import compose
    from .source_domain import compose_source_domain
    from .domain_transport import write_domain_catalog

    if runtime_services.settings.JARVIS_OWNER_ID != settings.JARVIS_OWNER_ID:
        raise ValueError("Runtime services belong to another owner")
    recording = getattr(runtime_services, "recording_source_access", None)
    candidates = ([recording.files] if recording is not None else [module.repository
        for module in runtime_services.tool_modules if isinstance(module, FileToolModule)
        and isinstance(module.repository, SourceFileRepository)])
    if len(candidates) != 1 or candidates[0].sources.pool is not runtime_services.db_pool:
        raise ValueError("Retained file tools require one initialized owned source repository")
    files = candidates[0]
    requests = (recording.tasks if recording is not None else
                TaskRepository(runtime_services.db_pool, TaskLimits()))
    client = HermesGatewayClient(home)
    bridge = compose(home, client)
    runtime_slot = {}
    operational = getattr(runtime_services, "companion_tools", None)
    if read_live_state is None and operational is not None:
        async def read_live_state(scope, arguments):
            if scope.owner_id != operational.owner_id:
                raise ValueError("Live State owner mismatch")
            return await operational.state.read_current()
    extra = ({"additional_modules": (operational,),
              "additional_tools": operational.names(),
              "additional_retained_writes": operational.names()}
             if operational is not None else {})
    source = compose_source_domain(bridge.service, scope_for_input=bridge.scopes.scope_for_input,
        sources=files.sources, requests=requests, storage=files.storage, area_id=files.area_id,
        selections=files.selections, access_check=files.access_check, settings=settings, files=files,
        sandbox_service=lambda: runtime_slot["runtime"].sandboxes, read_live_state=read_live_state,
        **extra)
    write_domain_catalog(home, source.catalog)

    async def prepare_event_scope(*, session_id, native_input_id, event):
        previous = await asyncio.to_thread(bridge.scopes.latest_for_session, session_id)
        return DomainInputScope(settings.JARVIS_OWNER_ID, session_id, native_input_id,
            previous.jarvis_session_id, frozenset(previous.allowed_tools & source.allowed_tools))

    async def prepare_voice_input(*, session_id, input_id, jarvis_session_id):
        binding = bridge.service.bindings.get(session_id)
        if not binding.is_master or binding.native_id is None:
            raise ValueError("Voice input must target the established retained master")
        return await bridge.service.bind_input_scope(session_id, input_id=input_id,
            jarvis_session_id=jarvis_session_id, allowed_tools=source.allowed_tools,
            request_event_id=None, request_revision=None)

    runtime = RetainedProfileRuntime(home, **options, domain_adapter=source.adapter,
        prepare_event_scope=prepare_event_scope, client_factory=lambda path: client,
        bridge_factory=lambda path, selected_client: bridge)
    runtime_slot["runtime"] = runtime
    runtime.source_domain = source
    runtime.voice_allowed_tools = source.allowed_tools
    runtime.prepare_voice_input = prepare_voice_input
    return _connect_master_notices(runtime, runtime_services)
