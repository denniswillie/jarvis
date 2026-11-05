"""Real direct admission/capture/writer/dispatcher with controlled external I/O.

Canonical records are real DTOs over an in-memory persistence boundary, NOT SQL
qualification. Task tool handlers and their TaskVoiceSource/idempotency logic are
real; the final task service records controlled effects, never runs a worker.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.realtime.request_sources import REQUEST_SOURCE_ARGUMENT
from app.realtime.source_context import LiveResponseSources
from app.tasks.context_events import ContextRecord
from app.tasks.models import TaskError, TaskStatus, TaskVoiceSource
from app.tasks.voice_tools import BackgroundTaskToolModule
from app.tools.module import ToolBuildContext
import test_realtime_auto_vad_request_evidence as fixtures


class Sources:
    def __init__(self):
        self.revoked = set()
        self.queries = []

    @asynccontextmanager
    async def acquire(self):
        yield self

    @asynccontextmanager
    async def transaction(self):
        yield self

    async def fetchval(self, query, *args):
        self.queries.append(query)
        if "lock_jarvis_copy_sources" in query:
            return not self.revoked.intersection(args[1])
        if "insert into jarvis_context_payloads" in query:
            return uuid4()
        raise AssertionError(query)

    async def executemany(self, query, args):
        self.queries.append(query)


class CanonicalStore:
    def __init__(self, sources):
        self.sources, self.rows = sources, {}
        self.lookups = 0

    async def append(self, owner, event):
        key = owner, event.session_id, event.source_namespace, event.source_event_id
        old = self.rows.get(key)
        if old is not None:
            self.sources.revoked.add(old.payload_id)
        record = ContextRecord(id=uuid4(), owner_id=owner, session_id=event.session_id,
            sequence=len(self.rows) + 1, origin_sequence=old.origin_sequence if old else len(self.rows) + 1,
            source_namespace=event.source_namespace, source_event_id=event.source_event_id,
            revision=old.revision + 1 if old else 1, supersedes_event_id=old.id if old else None,
            kind=event.kind, trust=event.trust, finality=event.finality, source_time=event.source_time,
            recorded_at=datetime.now(UTC), payload_id=uuid4(), availability="available",
            content_digest=None, content=event.payload, is_current=True)
        self.rows[key] = record
        return record

    async def request_evidence(self, owner, session, *, source_namespace, source_event_id):
        self.lookups += 1
        record = self.rows.get((owner, session, source_namespace, source_event_id))
        if record is None or not record.authorizes_request:
            raise TaskError("task_request_evidence_missing", "Unavailable")
        return record

    def withdraw(self, record):
        key = record.owner_id, record.session_id, record.source_namespace, record.source_event_id
        self.rows[key] = record.model_copy(update={"availability": "removed", "is_current": False})
        self.sources.revoked.add(record.payload_id)


class WaitingRoute:
    def __init__(self):
        self.ready = True
        self.context_refresh_pending = True
        self.entered, self.gate = asyncio.Event(), asyncio.Event()

    async def wait_for_context_ready(self, timeout):
        self.entered.set()
        await self.gate.wait()
        return True

    def release(self):
        self.context_refresh_pending = False
        self.gate.set()


class ControlledTasks:
    def __init__(self, session):
        self.calls, self.effects = [], {}
        self.task = SimpleNamespace(id=uuid4(), revision=2, title="WorkbenchPlan", mode="background",
            status=TaskStatus.CANCEL_REQUESTED, pending_status="cancelled", reason="cancel_requested",
            session_id=session.context.jarvis_session_id, knowledge_area_id=None, updated_at=datetime.now(UTC))
        self.repository = SimpleNamespace(pool=None, reference=lambda _: "task_123456abcdef",
            resolve_reference=AsyncMock(return_value=self.task.id))

    def provider_configured(self, kind):
        return False

    async def mutate(self, owner, task, operation, request, *, source):
        assert isinstance(source, TaskVoiceSource)
        self.calls.append((operation, request, source))
        self.effects.setdefault(request.request_id, source)
        return self.task


@asynccontextmanager
async def composed():
    fx = fixtures.NativeAutoVADRequestEvidenceTests()
    await fx.asyncSetUp()
    fx.sources = Sources()
    fx.store = CanonicalStore(fx.sources)
    fx.capture.repository = fx.store
    fx.session._source_context = LiveResponseSources(fx.sources,
        owner_id=fx.session.context.owner_id, session_id=fx.session.context.jarvis_session_id,
        audio_session_id=fx.session.context.audio_session_id)
    fx.route = WaitingRoute()
    fx.session._voice_route_runtime = fx.route
    fx.tasks = ControlledTasks(fx.session)
    fx.session.tool_registry.register_many(BackgroundTaskToolModule(fx.tasks).build_tools(
        ToolBuildContext(None, fx.session.context.owner_id, fx.session.context.jarvis_session_id)))
    try:
        yield fx
    finally:
        fx.route.release()
        fx.session._voice_route_runtime = None
        await fx.asyncTearDown()


async def admitted(fx, item, text=None):
    await fx.start(item)
    await fx.event("input_audio_buffer.committed", item_id=item)
    if text is not None:
        await fx.transcript(item, text)


async def cancelled_waiter_then_current(fx, *, original_final=True, current_text="Separately, what is two plus three?"):
    await admitted(fx, "old-cancel")
    await asyncio.wait_for(fx.route.entered.wait(), 1)
    if original_final:
        await fx.transcript("old-cancel", "Cancel WorkbenchPlan. Do not announce its result.")
    old = fx.session._response_preparation_task
    await admitted(fx, "current-question", current_text)
    fx.route.release()
    await asyncio.wait_for(asyncio.gather(*tuple(fx.session._direct_tasks)), 2)
    assert old.done() and old.cancelled()
    assert len(fx.creates()) == 1
    write = fx.creates()[0]
    assert write["response"]["metadata"]["jarvis_turn_generation"] == "2"
    assert fx.store.lookups == 0  # No new per-turn canonical query.
    await fx.event("response.created", response={"id": "response-current", "metadata": write["response"]["metadata"]})
    state = fx.session._active_responses["response-current"]
    assert not state.interrupted
    return write, state


async def call(fx, name, call_id, args):
    await fx.event("response.output_item.done", response_id="response-current", item={
        "id": call_id + "-item", "type": "function_call", "call_id": call_id,
        "name": name, "arguments": json.dumps(args)})
    await fx.until(lambda: not fx.session._ordinary_tool_tasks)
    outputs = [event["item"] for event in fx.provider.events
               if event.get("type") == "conversation.item.create"
               and event.get("item", {}).get("call_id") == call_id]
    assert outputs
    return json.loads(outputs[-1]["output"])


def references(state):
    assert state.request_sources is not None
    return {r.kind: r for r in state.request_sources.requests}


def test_actual_unwritten_waiter_carries_exact_sources_and_real_task_tool_attribution():
    async def run():
        async with composed() as fx:
            write, state = await cancelled_waiter_then_current(fx)
            refs = references(state)
            assert "Cancel WorkbenchPlan" in write["response"]["instructions"]
            assert "two plus three" in write["response"]["instructions"]
            schema = next(t for t in write["response"]["tools"] if t["name"] == "cancel_background_task")
            assert set(schema["parameters"]["properties"][REQUEST_SOURCE_ARGUMENT]["enum"]) == {
                r.reference for r in refs.values()}
            assert REQUEST_SOURCE_ARGUMENT not in fx.session.tool_registry.get("cancel_background_task").input_schema["properties"]
            assert set(state.source_snapshot.sources) >= set(state.request_sources.roots)
            args = dict(task_ref="task_123456abcdef", expected_revision=1,
                        jarvis_request_source=refs["retained_unwritten"].reference)
            result = await call(fx, "cancel_background_task", "cancel-call", args)
            assert result["status"] == "success"
            _, request, source = fx.tasks.calls[0]
            assert source.event_id == refs["retained_unwritten"].record.id
            assert source.revision == refs["retained_unwritten"].record.revision
            assert state.tool_followup.request.event_id == source.event_id
            assert state.tool_followup.conversation_request.event_id == refs["current"].record.id
            # Duplicate provider callback executes nothing. A distinct provider
            # call for the SAME source retains actual task-operation identity.
            await call(fx, "cancel_background_task", "cancel-call", args)
            assert len(fx.tasks.calls) == 1
            await call(fx, "cancel_background_task", "cancel-retry", args)
            assert fx.tasks.calls[-1][1].request_id == request.request_id
            assert len(fx.tasks.effects) == 1
            result = await call(fx, "fixture_request_evidence", "current-call",
                {REQUEST_SOURCE_ARGUMENT: refs["current"].reference, "value": 5})
            assert result["status"] == "success"
            assert fx.invocations[-1][0].request_event_id == refs["current"].record.id
            assert fx.invocations[-1][1] == {"value": 5}
            assert not state.tool_followup.binding_conflict
            assert fx.session._turn_generation == 2
            assert not fx.capture._unwritten_inputs  # Offered, NOT marked fulfilled.
    asyncio.run(run())


@pytest.mark.parametrize("selection", [None, "request_unknown", "", 3, ["request_unknown"]])
def test_missing_foreign_or_malformed_selection_fails_before_effects(selection):
    async def run():
        async with composed() as fx:
            await cancelled_waiter_then_current(fx)
            args = dict(task_ref="task_123456abcdef", expected_revision=1)
            if selection is not None:
                args[REQUEST_SOURCE_ARGUMENT] = selection
            result = await call(fx, "cancel_background_task", "rejected-call", args)
            assert result["status"] == "failed"
            assert result["error"]["code"] in {"tool_request_source_required", "tool_request_source_invalid"}
            assert not fx.tasks.calls and fx.store.lookups == 0
    asyncio.run(run())


@pytest.mark.parametrize("change", ["revision", "withdrawal", "owner", "epoch", "provider"])
def test_selected_original_must_still_match_authority(change):
    async def run():
        async with composed() as fx:
            _, state = await cancelled_waiter_then_current(fx)
            source = references(state)["retained_unwritten"]
            if change == "revision":
                await fx.transcript("old-cancel", "Correction: do not cancel it.")
            elif change == "withdrawal":
                fx.store.withdraw(source.record)
            elif change == "owner":
                fx.session.context = replace(fx.session.context, owner_id="foreign")
            elif change == "epoch":
                fx.session._visual_context.connection = replace(fx.session._visual_context.connection, connection_epoch=uuid4())
            else:
                fx.session.openai = fixtures.NativeProvider()
            # Real dispatcher selection + executor, without retrying output from
            # a deliberately invalidated transport/source fixture.
            from app.assistant.tool_limits import ToolCallBudget
            result = await fx.session._select_tool_result(name="cancel_background_task",
                arguments_text=json.dumps(dict(task_ref="task_123456abcdef", expected_revision=1,
                    jarvis_request_source=source.reference)), call_id="changed-call", source_state=state,
                budget=ToolCallBudget(max_total_calls=20, max_repeated_calls=5))
            assert result["status"] == "failed"
            assert not fx.tasks.calls
    asyncio.run(run())


def test_late_retained_final_does_not_change_already_written_offer_or_execute():
    async def run():
        async with composed() as fx:
            write, state = await cancelled_waiter_then_current(fx, original_final=False)
            source = references(state)["retained_unwritten"]
            frozen = json.dumps(write)
            assert source.record is None
            await fx.transcript("old-cancel", "Cancel WorkbenchPlan.")
            result = await call(fx, "cancel_background_task", "late-call",
                dict(task_ref="task_123456abcdef", expected_revision=1, jarvis_request_source=source.reference))
            assert result["status"] == "failed" and not fx.tasks.calls
            assert json.dumps(write) == frozen and len(fx.creates()) == 1
            assert fx.capture._unwritten_inputs  # Never called handled by speech.
    asyncio.run(run())


def test_latest_correction_can_be_selected_without_executing_older_cancel():
    async def run():
        async with composed() as fx:
            write, state = await cancelled_waiter_then_current(fx,
                current_text="Correction: do not cancel WorkbenchPlan. Keep working. What is two plus three?")
            refs = references(state)
            result = await call(fx, "fixture_request_evidence", "newest-only",
                {REQUEST_SOURCE_ARGUMENT: refs["current"].reference})
            assert result["status"] == "success"
            assert fx.invocations[-1][0].request_event_id == refs["current"].record.id
            assert not fx.tasks.calls and "Newer explicit corrections" in write["response"]["instructions"]
            # Controlled source choice, not a measured model-intent classifier.
    asyncio.run(run())


@pytest.mark.parametrize("unknown", [False, True])
def test_written_or_unknown_response_is_not_carried_as_unwritten(unknown):
    async def run():
        async with composed() as fx:
            fx.route.release()
            original = fx.provider.send_event
            async def transport(event, **kwargs):
                result = await original(event, **kwargs)
                if unknown and fx.provider.events[-1]["type"] == "response.create":
                    raise RuntimeError("controlled write outcome unknown")
                return result
            fx.provider.send_event = transport
            await admitted(fx, "already-written", "Cancel WorkbenchPlan.")
            results = await asyncio.gather(*tuple(fx.session._direct_tasks), return_exceptions=True)
            assert len(fx.creates()) == 1
            assert not fx.capture._unwritten_inputs
            fx.provider.send_event = original
            await admitted(fx, "next-input", "What is two plus three?")
            await asyncio.gather(*tuple(fx.session._direct_tasks), return_exceptions=True)
            assert not fx.capture._unwritten_inputs
            assert REQUEST_SOURCE_ARGUMENT not in json.dumps(fx.creates()[-1])
            assert not fx.tasks.calls
    asyncio.run(run())


def test_single_request_remains_compatible_and_reset_clears_retained_refs():
    async def run():
        async with composed() as fx:
            fx.route.release()
            await admitted(fx, "single", "Inspect my request.")
            await asyncio.gather(*tuple(fx.session._direct_tasks))
            write = fx.creates()[0]
            await fx.event("response.created", response={"id": "response-current", "metadata": write["response"]["metadata"]})
            result = await call(fx, "fixture_request_evidence", "single-call", {"value": 5})
            assert result["status"] == "success" and len(fx.invocations) == 1
            assert fx.invocations[0][0].request_event_id is not None
            assert REQUEST_SOURCE_ARGUMENT not in json.dumps(write)
            from app.realtime.request_sources import RequestSourceScope
            fx.capture.retain_unwritten(1, RequestSourceScope.from_session(fx.session))
            assert fx.capture._unwritten_inputs
            fx.capture.context_reset("reset", client_cleared=False, input_was_pending=False)
            assert not fx.capture._unwritten_inputs
    asyncio.run(run())
