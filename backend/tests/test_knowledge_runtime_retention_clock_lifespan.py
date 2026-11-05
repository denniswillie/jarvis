"""Actual app lifespan and knowledge factory composition, with offline IO seams."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from app import main
from app.knowledge import runtime as knowledge
from app.settings import JarvisSettings
from test_knowledge_runtime_retention_clock_factory import RetentionClock, factory


REAL_KNOWLEDGE_START = knowledge.KnowledgeRuntime.start


@pytest.fixture
def application(factory, monkeypatch):
    settings = JarvisSettings(
        _env_file=None, APP_ENV="test", JARVIS_OWNER_ID="factory-owner",
        JARVIS_STATIC_BEARER_TOKEN="offline-lifespan-private-token",
        SUPABASE_DB_URL=None, OPENAI_API_KEY=None, GEMINI_API_KEY=None,
        VISUAL_REALTIME_ENABLED=False, VISUAL_FAST_PERCEPTION_ENABLED=False,
        JARVIS_RESTORE_IN_PROGRESS=False,
    )
    factory.pool.close = AsyncMock()
    events, runtimes = [], []

    async def initialize_base(services):
        events.append("base")
        services.db_pool = factory.pool

    async def initialize_tasks(services):
        events.append("tasks")

    def start(runtime):
        # The actual factory has constructed the real retention service already.
        events.append("knowledge_start")
        runtimes.append(runtime)
        if isinstance(runtime.retention._clock, RetentionClock):
            assert runtime.retention._clock.calls == 0

        async def ingestion(owner, worker_id, stop, **kwargs):
            # Run the normal lifecycle without doing external ingestion work.
            assert owner == "factory-owner"
            await stop.wait()

        runtime.worker.run = ingestion
        REAL_KNOWLEDGE_START(runtime)

    def select_memory(services):
        events.append("selected_memory")

    async def initialize_attention(services):
        events.append("attention")

    async def initialize_reminders(services):
        events.append("reminders")

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "configure_logging", Mock())
    monkeypatch.setattr(main, "_initialize_runtime_services", initialize_base)
    monkeypatch.setattr(main, "initialize_task_runtime", initialize_tasks)
    monkeypatch.setattr(main, "configure_selected_memory", select_memory)
    monkeypatch.setattr(main, "initialize_attention_runtime", initialize_attention)
    monkeypatch.setattr(main, "_initialize_reminder_services", initialize_reminders)
    monkeypatch.setattr(knowledge.KnowledgeRuntime, "start", start)
    return SimpleNamespace(factory=factory, settings=settings, events=events, runtimes=runtimes)


def assert_runtime_joined(runtime):
    assert runtime._closed
    assert runtime.task.done()
    assert runtime.file_task.done()
    assert runtime.retention_task.done()
    runtime.client.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["omitted", "none", "custom"])
async def test_actual_app_lifespan_reaches_retention_before_start_and_joins(application, mode):
    clock = RetentionClock() if mode == "custom" else None
    kwargs = {} if mode == "omitted" else {"retention_clock": clock}
    app = main.create_app(**kwargs)
    async with app.router.lifespan_context(app):
        services = app.state.runtime_services
        runtime = services.knowledge_runtime
        assert isinstance(runtime, knowledge.KnowledgeRuntime)
        assert runtime.retention._clock is clock
        assert runtime.context_repository.tasks.pool is application.factory.pool
        assert runtime.task is not None and runtime.retention_task is not None
        assert services.current_image_service._max_age_seconds == application.settings.VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS
        assert services.settings is application.settings
        assert application.events == ["base", "tasks", "knowledge_start", "selected_memory", "attention", "reminders"]
        assert runtime.retention._clock_time() == (clock.value if clock is not None else None)
    assert_runtime_joined(runtime)
    application.factory.pool.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_default_lifespan_keeps_initializer_call_shape(application, monkeypatch):
    initialize = AsyncMock(wraps=main.initialize_knowledge_runtime)
    monkeypatch.setattr(main, "initialize_knowledge_runtime", initialize)
    app = main.create_app()
    async with app.router.lifespan_context(app):
        services = app.state.runtime_services
        initialize.assert_awaited_once_with(services)
    assert_runtime_joined(services.knowledge_runtime)


@pytest.mark.asyncio
async def test_application_bindings_are_isolated_not_a_process_clock(application):
    clock = RetentionClock()
    custom = main.create_app(retention_clock=clock)
    ordinary = main.create_app()
    for app, expected in ((custom, clock), (ordinary, None)):
        async with app.router.lifespan_context(app):
            runtime = app.state.runtime_services.knowledge_runtime
            assert runtime.retention._clock is expected
        assert_runtime_joined(runtime)
    assert clock.calls == 0
    assert application.factory.pool.close.await_count == 2


@pytest.mark.asyncio
async def test_request_environment_and_state_cannot_select_retention_clock(application, monkeypatch):
    trusted, untrusted = RetentionClock(), RetentionClock()
    monkeypatch.setenv("JARVIS_RETENTION_CLOCK", "untrusted-clock-value")
    app = main.create_app(retention_clock=trusted)
    app.state.retention_clock = untrusted
    async with app.router.lifespan_context(app):
        runtime = app.state.runtime_services.knowledge_runtime
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await client.get("/health?retention_clock=untrusted-clock-value", headers={"x-retention-clock": "untrusted-clock-value"})
        assert runtime.retention._clock is trusted
        assert trusted.calls == untrusted.calls == 0
    assert_runtime_joined(runtime)


@pytest.mark.asyncio
async def test_later_startup_failure_joins_real_knowledge_tasks(application, monkeypatch):
    clock = RetentionClock()

    def fail_after_knowledge(services):
        assert services.knowledge_runtime.retention._clock is clock
        raise RuntimeError("offline_selected_memory_failure")

    monkeypatch.setattr(main, "configure_selected_memory", fail_after_knowledge)
    app = main.create_app(retention_clock=clock)
    with pytest.raises(RuntimeError, match="offline_selected_memory_failure"):
        async with app.router.lifespan_context(app):
            pytest.fail("startup failure must precede admission")
    assert len(application.runtimes) == 1
    assert_runtime_joined(application.runtimes[0])
    application.factory.pool.close.assert_awaited_once()
    assert clock.calls == 0


@pytest.mark.asyncio
async def test_disabled_knowledge_does_not_sample_optional_clock(application):
    application.factory.resolution.enabled = False
    clock = RetentionClock()
    app = main.create_app(retention_clock=clock)
    async with app.router.lifespan_context(app):
        assert app.state.runtime_services.knowledge_runtime is None
    assert application.runtimes == application.factory.clients == []
    assert clock.calls == 0
    application.factory.pool.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_restore_guard_still_precedes_runtime_initialization(application, monkeypatch):
    settings = application.settings.model_copy(update={"JARVIS_RESTORE_IN_PROGRESS": True})
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    clock = RetentionClock()
    app = main.create_app(retention_clock=clock)
    with pytest.raises(RuntimeError, match="jarvis_restore_in_progress"):
        async with app.router.lifespan_context(app):
            pytest.fail("restore guard must precede admission")
    assert application.events == application.runtimes == []
    application.factory.pool.close.assert_not_awaited()
    assert clock.calls == 0


@pytest.mark.parametrize("invalid", [0, "2030-01-01", object()])
def test_noncallable_dependency_rejected_at_app_composition(application, invalid):
    with pytest.raises(ValueError, match="Retention clock must be callable"):
        main.create_app(retention_clock=invalid)
    assert application.events == application.factory.clients == []
