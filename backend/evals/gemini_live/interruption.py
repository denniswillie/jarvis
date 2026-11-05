"""Opt-in interruption fixtures using Jarvis's actual non-image tool lifecycle."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import asyncpg

from app.experience.repository import ToolExecutionRepository
from app.memory.repository import MemoryRepository
from app.memory.service import MemoryApplicationService
from app.memory.tools import MemoryToolModule
from app.perception.frame_window import RecentVisualFrameWindow
from app.realtime.session import RealtimeAudioSession
from app.sessions.repository import SessionRepository
from app.tools.executor import JarvisToolExecutor
from app.tools.module import ToolBuildContext
from app.tools.registry import JarvisToolRegistry
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners


STAGES = ("before_admission", "during_execution", "after_commit", "after_receipt")
OWNER_PREFIX = "direct-image-eval-interruption-"


def tool_result_succeeded(result: object) -> bool:
    """A returned validation error is not a successful mutation checkpoint."""
    return isinstance(result, dict) and result.get("status") == "success"


class Trace:
    def __init__(self):
        self.started = time.monotonic()
        self.events: list[dict] = []

    def add(self, event: str, **fields):
        self.events.append({"seconds": round(time.monotonic() - self.started, 6),
                            "event": event, **fields})


class Barrier:
    """Hold one first operation at a named boundary, never an arbitrary sleep."""

    def __init__(self, stage: str | None, trace: Trace):
        if stage is not None and stage not in STAGES:
            raise ValueError("Unknown interruption stage")
        self.stage, self.trace = stage, trace
        self.reached, self.release = asyncio.Event(), asyncio.Event()
        self.claimed = False

    async def pause(self, stage: str):
        if stage != self.stage or self.claimed:
            return
        self.claimed = True
        self.trace.add("barrier_reached", stage=stage)
        self.reached.set()
        await self.release.wait()
        self.trace.add("barrier_released", stage=stage)


class ObservedExecutor:
    def __init__(self, executor, barrier: Barrier, trace: Trace):
        self.executor, self.barrier, self.trace = executor, barrier, trace

    async def execute(self, name, arguments, **kwargs):
        self.trace.add("execution_admitted", call_id=kwargs.get("provider_call_id"))
        result = await self.executor.execute(name, arguments, **kwargs)
        self.trace.add("execution_returned", call_id=kwargs.get("provider_call_id"),
                       result=result)
        await self.barrier.pause("after_commit")
        return result


def instrument_registry(registry, barrier: Barrier, trace: Trace):
    observed = JarvisToolRegistry()
    for definition in registry.list():
        async def handler(arguments, original=definition.handler):
            self_call = __import__("app.tools.invocation", fromlist=["current_tool_invocation"])
            invocation = self_call.current_tool_invocation()
            trace.add("handler_entered", call_id=invocation.provider_call_id)
            # The durable execution row is already started here. This is before
            # the real memory service opens its mutation transaction, not mid-SQL.
            await barrier.pause("during_execution")
            result = await original(arguments)
            trace.add("handler_returned", call_id=invocation.provider_call_id,
                      result=result)
            return result
        observed.register(replace(definition, handler=handler))
    return observed


class Lifecycle:
    """Run production admission/execution methods with a test-only transport.

    Only application notifications, pending-call settlement, and provider-specific
    response scheduling are outside this slice. No image path or iOS playback is
    represented. Gemini origin IDs are explicitly client-assigned turn IDs.
    """

    def __init__(self, registry, executor, send_output, barrier, trace):
        self.barrier, self.trace = barrier, trace
        session = object.__new__(RealtimeAudioSession)
        session.settings = SimpleNamespace(JARVIS_MAX_TOOL_CALLS_PER_RESPONSE=8,
                                           JARVIS_MAX_REPEATED_TOOL_CALLS_PER_RESPONSE=4)
        session.tool_registry = registry
        session.tool_executor = ObservedExecutor(executor, barrier, trace)
        session._visual_interrupt_control_registry = JarvisToolRegistry()
        session._active_responses = {}
        session._current_response_id = None
        session._turn_generation = 0
        session._handled_tool_call_ids = set()
        session._tool_chain_by_response_id = {}
        session._tool_budgets = {}
        session._responses_with_tool_calls = set()
        session._closed = asyncio.Event()
        session._response_admission_lock = asyncio.Lock()
        session._normal_turn_pending = False
        session._tool_followup = None
        session.openai = SimpleNamespace(send_event=send_output)
        session._tool_call_settled = lambda state, call_id: state.pending_tool_call_ids.discard(call_id)
        self.session = session

    @property
    def generation(self):
        return self.session._turn_generation

    def source(self, response_id: str):
        state = self.session._active_responses.get(response_id)
        if state is None:
            state = SimpleNamespace(
                response_id=response_id, interrupted=False, dismissed=False,
                completed_response=None, pending_tool_call_ids=set(),
                delivery_kind="normal", task_instructions=None,
                requested_image_call_ids=[], delivery_continuation_metadata=lambda: {},
            )
            self.session._active_responses[response_id] = state
        return state

    def interrupt(self):
        self.session._turn_generation += 1
        for state in self.session._active_responses.values():
            state.interrupted = True
        self.session._tool_followup = None
        self.session._normal_turn_pending = False
        self.trace.add("application_interruption", generation=self.generation)

    async def dispatch(self, call: dict, response_id: str):
        call_id, name, arguments = call.get("id"), call.get("name"), call.get("args", {})
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not isinstance(arguments, dict):
            raise ValueError("Malformed provider tool call")
        state = self.source(response_id)
        generation = self.generation
        self.trace.add("tool_call_received", call_id=call_id, name=name,
                       arguments=arguments, response_id=response_id, generation=generation)
        await self.barrier.pause("before_admission")
        await self.session._handle_function_call_item(
            {"call_id": call_id, "name": name, "arguments": json.dumps(arguments)},
            response_id=response_id,
        )
        if call_id not in self.session._handled_tool_call_ids:
            self.trace.add("tool_call_not_admitted", call_id=call_id)
        return not state.interrupted and generation == self.generation


class DatabaseFixtures:
    """Real commits, disposable owners, independent readback, mandatory cleanup."""

    def __init__(self, dsn: str, output: Path):
        if not dsn:
            raise ValueError("An explicit database DSN is required")
        self.dsn, self.output = dsn, output
        self.owners: list[str] = []
        self.pool = None
        self.cleanup_result = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=3,
                                            timeout=15, command_timeout=20)

    def _manifest(self, **extra):
        payload = {"owners": self.owners, "cleanup": self.cleanup_result, **extra}
        temporary = self.output / "cleanup-manifest.tmp"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.output / "cleanup-manifest.json")

    async def create(self, label: str):
        owner = OWNER_PREFIX + uuid4().hex
        self.owners.append(owner)
        self._manifest(state="fixtures_active")  # Before the first database write.
        async with self.pool.acquire() as connection:
            session_id = await connection.fetchval(
                "insert into jarvis_sessions(owner_id, title) values($1, $2) returning id",
                owner, "Disposable interruption evaluation: " + label,
            )
        return owner, session_id

    async def message(self, owner, session_id, content):
        await SessionRepository(self.pool).create_message(
            owner_id=owner, session_id=session_id, role="user", modality="text", content=content,
        )

    def registry(self, owner, session_id, *, names=("remember",)):
        repository = MemoryRepository(self.pool)
        service = MemoryApplicationService(
            repository=repository, retrieval_service=SimpleNamespace(),
            session_repository=SessionRepository(self.pool),
            frame_window=RecentVisualFrameWindow(max_frames_per_session=4, ttl_seconds=5),
            working_memory_service=None, embedding_job_repository=None,
            current_frame_max_age_seconds=3,
        )
        module = MemoryToolModule(repository=repository, entity_repository=SimpleNamespace(), service=service)
        registry = JarvisToolRegistry()
        # Real schema and real handler. Only this capability is under test; no
        # synthetic search/profile implementation or embedding provider is used.
        for tool in module.build_tools(ToolBuildContext(None, owner, session_id)):
            if tool.name in names:
                registry.register(tool)
        return registry

    def executor(self, registry, owner, session_id):
        return JarvisToolExecutor(registry, execution_repository=ToolExecutionRepository(self.pool),
                                  owner_id=owner, session_id=session_id)

    async def snapshot(self, owner, session_id):
        if owner not in self.owners:
            raise ValueError("Refusing to read a non-fixture owner")
        # A separate connection proves visibility after a real commit, rather
        # than reading our own uncommitted transaction or trusting a tool result.
        connection = await asyncpg.connect(self.dsn, timeout=15, command_timeout=20)
        try:
            memories = await connection.fetch(
                "select id, status, content, metadata from memory_items "
                "where owner_id=$1 and session_id=$2 order by created_at, id", owner, session_id,
            )
            executions = await connection.fetch(
                "select id, provider_call_id, tool_name, mutates_state, arguments, status, result, error, "
                "started_at, completed_at from jarvis_tool_executions "
                "where owner_id=$1 and session_id=$2 order by started_at, id", owner, session_id,
            )
            return {"memories": [dict(row) for row in memories],
                    "executions": [dict(row) for row in executions]}
        finally:
            await connection.close()

    async def cleanup(self):
        if self.pool is not None:
            try:
                await asyncio.wait_for(self.pool.close(), 15)
            except TimeoutError:
                self.pool.terminate()
            self.pool = None
        try:
            self.cleanup_result = (await cleanup_fixture_owners(self.dsn, set(self.owners))
                                   if self.owners else {"remaining_rows": 0, "owners": []})
        except BaseException as error:
            self._manifest(state="cleanup_failed", error_type=type(error).__name__)
            raise
        self._manifest(state="cleaned")
