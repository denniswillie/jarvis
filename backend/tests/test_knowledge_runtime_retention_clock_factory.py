"""Offline factory wiring only; not SQL expiry or logical-month acceptance."""
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from pydantic import SecretStr

from app.knowledge import runtime as module


class RetentionClock:
    def __init__(self):
        self.value = datetime(2026, 11, 1, tzinfo=UTC)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


@pytest.fixture
def factory(monkeypatch):
    configuration = module.KnowledgeConfiguration(
        base_url="https://memory.invalid",
        api_key=SecretStr("factory-only-not-a-real-credential-0123456789"),
        area_id=UUID("7336a161-4f59-4a13-acd5-8565921319bb"),
    )
    connection = SimpleNamespace(fetchval=AsyncMock(return_value="factory_fixture"))

    class Pool:
        @asynccontextmanager
        async def acquire(self):
            yield connection

    pool = Pool()
    clients, starts, source_owners = [], [], []

    class Client:
        def __init__(self, *args, **kwargs):
            self.arguments = args
            self.options = kwargs
            self.close = AsyncMock()
            clients.append(self)

    @asynccontextmanager
    async def transaction(sources, owner_id):
        assert sources.pool is pool
        source_owners.append(owner_id)
        yield connection

    def start(runtime):
        # Observe the real constructed service before any background task starts.
        starts.append(runtime.retention._clock)

    resolution = SimpleNamespace(
        enabled=True, configured=True, issue=None, configuration=configuration,
    )
    resolve = SimpleNamespace(resolve=lambda *args, **kwargs: resolution)
    monkeypatch.setattr(module, "get_second_brain_settings", lambda: resolve)
    monkeypatch.setattr(module, "read_schema_readiness", AsyncMock(return_value=SimpleNamespace(ready=True)))
    monkeypatch.setattr(module.SourceRepository, "transaction", transaction)
    monkeypatch.setattr(module.SourceRepository, "_area", AsyncMock())
    monkeypatch.setattr(module, "HindsightClient", Client)
    monkeypatch.setattr(module.KnowledgeRuntime, "start", start)
    services = SimpleNamespace(
        knowledge_runtime=None, knowledge_issue=None, db_pool=pool,
        settings=SimpleNamespace(SUPABASE_DB_URL="offline-unused", JARVIS_OWNER_ID="factory-owner"),
        background_task_service=None, assistant_context_providers=(),
    )
    return SimpleNamespace(configuration=configuration, connection=connection, pool=pool,
        clients=clients, starts=starts, source_owners=source_owners, services=services,
        resolution=resolution, client_type=Client)


async def construct(factory, entrypoint, kwargs):
    if entrypoint == "constructor":
        return module.KnowledgeRuntime(
            owner_id="factory-owner", sources=module.SourceRepository(factory.pool),
            tasks=module.TaskRepository(factory.pool, module.TaskLimits()),
            client=factory.client_type(), configuration=factory.configuration, **kwargs,
        )
    if entrypoint == "create":
        return await module.KnowledgeRuntime.create(
            pool=factory.pool, owner_id="factory-owner", configuration=factory.configuration, **kwargs,
        )
    await module.initialize_knowledge_runtime(factory.services, **kwargs)
    return factory.services.knowledge_runtime


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["constructor", "create", "initialize"])
@pytest.mark.parametrize("clock_mode", ["omitted", "none", "custom"])
async def test_retention_clock_reaches_existing_service_without_sampling(factory, entrypoint, clock_mode):
    clock = RetentionClock() if clock_mode == "custom" else None
    kwargs = {} if clock_mode == "omitted" else {"retention_clock": clock}
    runtime = await construct(factory, entrypoint, kwargs)
    assert runtime is not None
    try:
        assert runtime.retention._clock is clock
        assert runtime.configuration is factory.configuration
        assert runtime.sources.pool is factory.pool
        assert runtime.context_repository.tasks.pool is factory.pool
        assert runtime.task is runtime.file_task is runtime.retention_task is None
        assert factory.starts == ([clock] if entrypoint == "initialize" else [])
        assert "clock" not in runtime.client.options
        assert "retention_clock" not in runtime.client.options
        if clock is None:
            # None is the existing retention service's database-time sentinel.
            assert runtime.retention._clock_time() is None
        else:
            assert clock.calls == 0
            assert runtime.retention._clock_time() == clock.value
            clock.value += timedelta(days=31)
            assert runtime.retention._clock_time() == clock.value
            assert clock.calls == 2
        if entrypoint == "initialize":
            assert factory.services.knowledge_issue is None
            assert len(factory.services.assistant_context_providers) == 1
    finally:
        await runtime.close()
    runtime.client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_clock_is_per_runtime_not_global(factory):
    clock = RetentionClock()
    custom = await construct(factory, "create", {"retention_clock": clock})
    ordinary = await construct(factory, "create", {})
    try:
        assert custom.retention._clock is clock
        assert ordinary.retention._clock is None
        assert ordinary.retention._clock_time() is None
        assert clock.calls == 0
    finally:
        await custom.close()
        await ordinary.close()
    assert all(client.close.await_count == 1 for client in factory.clients)


@pytest.mark.asyncio
async def test_initialize_preserves_existing_task_repository(factory):
    clock = RetentionClock()
    tasks = module.TaskRepository(factory.pool, module.TaskLimits())
    factory.services.background_task_service = SimpleNamespace(repository=tasks)
    await module.initialize_knowledge_runtime(factory.services, retention_clock=clock)
    runtime = factory.services.knowledge_runtime
    try:
        assert runtime.context_repository.tasks is tasks
        assert runtime.retention._clock is clock
        assert factory.source_owners == ["factory-owner"]
        assert clock.calls == 0
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["create", "initialize"])
async def test_invalid_clock_closes_constructed_client(factory, entrypoint):
    if entrypoint == "create":
        with pytest.raises(ValueError, match="Retention clock must be callable"):
            await construct(factory, entrypoint, {"retention_clock": object()})
    else:
        await module.initialize_knowledge_runtime(factory.services, retention_clock=object())
        assert factory.services.knowledge_runtime is None
        assert factory.services.knowledge_issue == "knowledge_initialization_failed"
        assert factory.services.assistant_context_providers == ()
    assert factory.starts == []
    assert len(factory.clients) == 1
    factory.clients[0].close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", ["disabled", "unconfigured", "database_missing", "configuration_invalid"])
async def test_initialize_admission_guards_do_not_consume_clock(factory, gate):
    clock = RetentionClock()
    if gate == "disabled":
        factory.resolution.enabled = False
    elif gate == "unconfigured":
        factory.resolution.configured = False
    elif gate == "database_missing":
        factory.services.db_pool = None
    else:
        factory.resolution.configuration = object()
    await module.initialize_knowledge_runtime(factory.services, retention_clock=clock)
    assert factory.services.knowledge_runtime is None
    assert factory.services.knowledge_issue is not None
    assert factory.clients == factory.starts == []
    assert clock.calls == 0
    factory.connection.fetchval.assert_not_awaited()


@pytest.mark.asyncio
async def test_initialized_runtime_is_not_rebound_to_a_later_clock(factory):
    first, later = RetentionClock(), RetentionClock()
    await module.initialize_knowledge_runtime(factory.services, retention_clock=first)
    runtime = factory.services.knowledge_runtime
    try:
        await module.initialize_knowledge_runtime(factory.services, retention_clock=later)
        assert factory.services.knowledge_runtime is runtime
        assert runtime.retention._clock is first
        assert factory.starts == [first]
        assert len(factory.clients) == 1
        assert first.calls == later.calls == 0
    finally:
        await runtime.close()
