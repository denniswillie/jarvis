"""Production composition for the isolated Live/Hermes acceptance sidecar.

Use --factory live_hermes_voice_runtime:application with the sibling runner.
This is NOT a FastAPI lifespan, new task engine or serving configuration. It
borrows exactly the manifested SQL fixture and starts no recording, perception,
Hindsight, sandbox, attention, or memory workers. The harness owns run/stop and
SQL cleanup. Model credentials stay in the parent; native Hermes uses its normal
authenticated loopback gateway.

Configuration: SecondBrainSettings reads JARVIS_VOICE_ACCEPTANCE_ENV_FILE or
backend/.env. JARVIS_HERMES_PYTHON may explicitly select the installed private
interpreter; otherwise the configured path or the existing local 0.19.0 runtime
is used. No interpreter is installed or downloaded here.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from app.assistant.profile_runtime import BehaviorProfileUpdates
from app.assistant.profile_service import BehaviorProfileService
from app.assistant.repository import BehaviorProfileRepository
from app.knowledge.models import SourceError
from app.perception.frame_window import RecentVisualFrameWindow
from app.realtime.live_session import create_live_session
from app.realtime.models import RealtimeSessionContext
from app.realtime.session_manager import RealtimeSessionManager
from app.runtime.concurrency import PriorityAsyncLimiter
from app.runtime.events import DomainEventBus
from app.runtime.feature_config import FrontierConfiguration, MemoryAuthority, SecondBrainSettings, TaskLimits
from app.runtime.services import JarvisRuntimeServices
from app.sessions.repository import SessionRepository
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.tasks.session_context import SessionTaskContextProvider, SessionTaskContextService
from app.tasks.worker import WorkerContextFactory
from app.visual_memory.latest_frame_buffer import LatestVisualFrameBuffer


ROOT = Path(__file__).resolve().parents[1]
LOCAL_HERMES = ROOT / ".jarvis-dev/hermes-native-0.19.0/bin/python"


def configuration(settings, limits, artifact_dir):
    """Configuration/path validation only: no SQL, native process or API calls."""
    limits.validate()
    flags = SecondBrainSettings(_env_file=os.environ.get("JARVIS_VOICE_ACCEPTANCE_ENV_FILE", ROOT / "backend/.env"))
    explicit = os.environ.get("JARVIS_HERMES_PYTHON")
    interpreter = Path(explicit or flags.hermes_python)
    if not explicit and str(interpreter) == "/opt/hermes/bin/python" and not interpreter.is_file():
        interpreter = LOCAL_HERMES
    if not interpreter.is_absolute() or not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise RuntimeError("acceptance_native_interpreter_unavailable")
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.is_absolute() or artifact_dir.is_symlink() or artifact_dir.stat().st_mode & 0o077:
        raise RuntimeError("acceptance_private_artifact_directory_required")
    frontier = FrontierConfiguration(model=flags.frontier_model,
        reasoning_effort=flags.frontier_reasoning_effort, api_key=settings.OPENAI_API_KEY)
    budgets = TaskLimits(**{name: getattr(limits, name) for name in (
        "active_seconds", "tool_steps", "request_output_tokens", "task_output_tokens", "task_input_tokens", "max_attempts")},
        owner_concurrency=1, global_concurrency=1, owner_queue_limit=3,
        owner_daily_tokens=128000, lease_seconds=30, poll_seconds=.2, shutdown_seconds=5)
    native = HermesRuntimeConfiguration(python=interpreter, work_root=artifact_dir / "hermes",
        host_id="live-voice-acceptance-" + uuid4().hex, max_iterations=min(20, limits.tool_steps),
        tool_timeout_seconds=30, stop_seconds=1)
    return frontier, budgets, native


def composition_receipt(path, *, fixture, session_id, session, runner, provider):
    value = {"format":"jarvis.live_hermes_voice_runtime.v1", "owner_id":fixture.owner,
        "session_id":str(session_id), "area_id":str(fixture.area),
        "session_factory":"app.realtime.live_session.create_live_session",
        "delegation_handler":type(session.delegation_handler).__name__,
        "worker_id":runner.worker_id, "worker_model":provider.model,
        "worker_reasoning_effort":provider.frontier.reasoning_effort,
        "native_python":str(provider.configuration.python), "limits":runner.repository.limits.model_dump(),
        "source_backed_account_note":False, "long_term_memory_worker":False,
        "recording":False, "sandbox":False, "paid_execution_qualified":False}
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "w") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


@asynccontextmanager
async def application(*, fixture, session_id, websocket, offer, client_capabilities,
                      limits, artifact_dir, settings):
    """Yield one registered, unstarted normal Live session and its scoped runner.

    The short SQL worker-readiness lease is scoped to this unique fixture owner
    and expires without renewal. The harness must start the yielded runner
    immediately; this lease is configuration/readiness, not native qualification.
    Canonical task/capture/source guards are production objects, not mocks.
    """
    frontier, budgets, native = configuration(settings, limits, artifact_dir)
    if fixture.sources.pool is not fixture.pool:
        raise RuntimeError("acceptance_source_pool_mismatch")
    selected = settings.model_copy(update={"JARVIS_OWNER_ID":fixture.owner,
        "OPENAI_LIVE_ENABLED":"true", "RESPONSIVE_VOICE_ENABLED":"true", "OPENAI_LIVE_MODEL":"gpt-live-1"})
    manager = RealtimeSessionManager()
    window = RecentVisualFrameWindow(max_frames_per_session=1, ttl_seconds=10)
    services = JarvisRuntimeServices(settings=selected, realtime_session_manager=manager,
        sensory_frame_window=window, latest_visual_frame_buffer=LatestVisualFrameBuffer(ttl_seconds=10, frame_window=window),
        domain_event_bus=DomainEventBus(), model_limiter=PriorityAsyncLimiter(1),
        memory_authority=MemoryAuthority.LEGACY, db_pool=fixture.pool)
    profiles = BehaviorProfileUpdates(manager)
    services.behavior_profile_updates = profiles
    services.behavior_profile_service = BehaviorProfileService(BehaviorProfileRepository(
        fixture.pool, on_committed=profiles.submit))

    repository = TaskRepository(fixture.pool, budgets)
    contexts = ContextRepository(repository)
    registry = TaskWorkerRegistry(fixture.pool)
    configured = {"frontier":ConfiguredTaskProvider(frontier.model)}
    services.background_task_service = TaskService(repository, configured, registry=registry)
    services.background_task_providers = configured
    services.background_task_context_service = SessionTaskContextService(contexts)
    services.assistant_context_providers = (SessionTaskContextProvider(services.background_task_context_service),)
    services.background_task_issue = None
    prepare = WorkerContextFactory(fixture.pool)
    provider = HermesTaskProvider(frontier, native, prepare=prepare)
    runner = TaskRunner(repository, {"frontier":provider}, owner_id=fixture.owner,
        worker_id="live-voice-acceptance-" + uuid4().hex, host_id=native.host_id, capabilities=prepare.capabilities)
    sessions = SessionRepository(fixture.pool)
    existing = await sessions.get_session(fixture.owner, session_id)
    if existing is None or existing.status != "active":
        raise RuntimeError("acceptance_active_owned_session_required")
    audio_id = str(uuid4())
    context = RealtimeSessionContext(audio_id, fixture.owner, session_id, "phone_audio", existing)
    capture = SessionContextCapture(contexts, owner_id=fixture.owner, session_id=session_id,
        provider="openai", audio_session_id=audio_id)
    session = None
    advertised = registered = False
    try:
        session = await create_live_session(websocket=websocket, context=context, settings=selected,
            repository=sessions, runtime_services=services, offer=offer, capture_mode="auto",
            client_capabilities=client_capabilities, context_capture=capture)
        if session.work is None or session.delegation_handler is not session.work:
            raise RuntimeError("acceptance_normal_live_work_bridge_missing")
        names = {tool["name"] for tool in session.work.commands.catalog()}
        if not {"read_live_state", "read_preferences", "reply_to_user"}.issubset(names):
            raise RuntimeError("acceptance_domain_tools_missing")
        registered = await manager.register(audio_session_id=audio_id, owner_id=fixture.owner,
            jarvis_session_id=session_id, session=session)
        if not registered:
            raise RuntimeError("acceptance_session_registration_failed")
        await registry.advertise(worker_id=runner.worker_id, owner_id=fixture.owner, host_id=native.host_id,
            providers=configured, ttl_seconds=min(300, limits.scenario_seconds + limits.cleanup_seconds + 30),
            capabilities=runner.capabilities)
        advertised = True
        composition_receipt(Path(artifact_dir) / "runtime-composition.json", fixture=fixture,
            session_id=session_id, session=session, runner=runner, provider=provider)
        yield SimpleNamespace(session=session, runner=runner, services=services)
    finally:
        # Do not call services.close(): the fixture, not this factory, owns pool
        # teardown. Keep canonical owners available until their dependents stop.
        try:
            async with asyncio.timeout(limits.cleanup_seconds):
                results = await asyncio.gather(*(call for call in (
                    session.close("test_cleanup") if session is not None else None,
                    runner.stop(), profiles.close()) if call is not None), return_exceptions=True)
                if services.live_work_runtime is not None:
                    await services.live_work_runtime.close()
                await capture.close()
                if any(isinstance(result, BaseException) for result in results):
                    raise RuntimeError("acceptance_owner_cleanup_failed")
        finally:
            try:
                if advertised:
                    await registry.withdraw(fixture.owner, runner.worker_id)
                if registered:
                    await manager.unregister(audio_session_id=audio_id, owner_id=fixture.owner, jarvis_session_id=session_id)
            finally:
                services._closed = True
