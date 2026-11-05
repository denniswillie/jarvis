"""Offline admission-to-followup wire qualification, not SQL/provider acceptance.

The direct session, tool dispatcher, source capture, task-tool handler, response
settlement, source guard and OpenAI writer are production code. Only canonical
storage, task effects, ambient readiness and the socket are controlled here.
"""
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4
import asyncio
import json

import pytest

from app.realtime.openai_client import OpenAIRealtimeConnection
from app.realtime.source_context import LiveResponseSources
from app.tasks.context_events import ContextRecord
from app.tasks.voice_tools import BackgroundTaskToolModule
import test_realtime_auto_vad_request_evidence as native_fixture


OLD_TEXT = "Cancel the background task named Workbench Plan. Do not announce its result."
CURRENT_TEXT = "What is two plus three? Do not restart Workbench Plan."
CORRECTION_TEXT = "Correction: do not cancel Workbench Plan. Only answer two plus three."
SELECTOR = "jarvis_request_source"


class CanonicalBoundary:
    """Exact DTOs and revocable roots; deliberately not a SQL simulation claim."""

    def __init__(self):
        self.records = {}
        self.unavailable = set()
        self.dependencies = {}
        self.locks = []
        self.depth = 0
        self.sequence = 0
        self.lookups = 0

    @asynccontextmanager
    async def acquire(self):
        yield self

    @asynccontextmanager
    async def transaction(self):
        self.depth += 1
        try:
            yield self
        finally:
            self.depth -= 1

    def is_in_transaction(self):
        return self.depth > 0

    async def fetchval(self, query, *args):
        if "lock_jarvis_copy_sources" in query:
            self.locks.append((args[0], tuple(args[1])))
            return all(root not in self.unavailable for root in args[1])
        if "insert into jarvis_context_payloads" in query:
            return uuid4()
        raise AssertionError("unexpected controlled persistence operation")

    async def executemany(self, query, rows):
        assert "jarvis_context_payload_dependencies" in query
        for owner, source, dependent in rows:
            self.dependencies.setdefault(dependent, set()).add(source)

    async def append(self, owner, ingress):
        key = (owner, ingress.session_id, ingress.source_namespace, ingress.source_event_id)
        old = self.records.get(key)
        self.sequence += 1
        if old is not None:
            self.unavailable.add(old.payload_id)
        record = ContextRecord(
            id=uuid4(), owner_id=owner, session_id=ingress.session_id,
            sequence=self.sequence, origin_sequence=old.origin_sequence if old else self.sequence,
            source_namespace=ingress.source_namespace, source_event_id=ingress.source_event_id,
            revision=old.revision + 1 if old else 1,
            supersedes_event_id=old.id if old else None,
            kind=ingress.kind, trust=ingress.trust, finality=ingress.finality,
            source_time=ingress.source_time, recorded_at=datetime.now(UTC),
            payload_id=uuid4(), availability="available", content_digest=None,
            content=ingress.payload, is_current=True,
        )
        self.records[key] = record
        return record

    async def request_evidence(self, owner, session_id, *, source_namespace, source_event_id):
        self.lookups += 1
        record = self.records[(owner, session_id, source_namespace, source_event_id)]
        if record.payload_id in self.unavailable:
            return record.model_copy(update={"availability": "removed", "is_current": False})
        return record

    def input(self, item_id):
        matches = [r for r in self.records.values()
                   if r.kind == "user_transcript" and r.content.get("provider_item_id") == item_id]
        assert len(matches) == 1
        return matches[0]


class ContextReadyBoundary:
    def __init__(self):
        self.release = asyncio.Event()
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.request = SimpleNamespace(route="controlled-direct-route")
        self.coordinator = SimpleNamespace(status=SimpleNamespace(phase="active", reason=None))

    @property
    def ready(self):
        return self.release.is_set()

    @property
    def context_refresh_pending(self):
        return not self.ready

    async def wait_for_context_ready(self, timeout):
        self.entered.set()
        try:
            async with asyncio.timeout(timeout):
                await self.release.wait()
            return True
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class SocketBoundary:
    def __init__(self, canonical):
        self.canonical = canonical
        self.events = []
        self.receipts = []
        self.fail_next_create = False

    async def send(self, encoded):
        event = json.loads(encoded)
        self.events.append(event)
        self.receipts.append({
            "event": event, "inside_source_transaction": self.canonical.depth > 0,
            "last_source_lock": self.canonical.locks[-1] if self.canonical.locks else None,
        })
        if event["type"] == "response.create" and self.fail_next_create:
            self.fail_next_create = False
            raise RuntimeError("controlled unknown write after socket handoff")

    async def close(self):
        pass


class ControlledTaskEffects:
    def __init__(self, h, canonical):
        self.calls = []
        self.effects = set()
        self.id = uuid4()
        self.ref = "task_" + self.id.hex[:12]
        self.repository = SimpleNamespace(pool=canonical,
            resolve_reference=self.resolve, reference=lambda task_id: "task_" + task_id.hex[:12])
        self.task = SimpleNamespace(id=self.id, revision=2, title="Workbench Plan", mode="background",
            status=SimpleNamespace(value="cancel_requested"), reason="user_cancelled", pending_status="cancelled",
            session_id=h.session.context.jarvis_session_id, knowledge_area_id=None, updated_at=datetime.now(UTC))

    def provider_configured(self, kind):
        return False

    async def resolve(self, owner, reference):
        assert reference == self.ref
        return self.id

    async def mutate(self, owner, task_id, operation, mutation, *, source):
        self.calls.append((owner, task_id, operation, mutation, source))
        self.effects.add(mutation.request_id)
        return self.task


@asynccontextmanager
async def wire_case():
    h = native_fixture.NativeAutoVADRequestEvidenceTests("runTest")
    await h.asyncSetUp()
    canonical = CanonicalBoundary()
    route = ContextReadyBoundary()
    socket = SocketBoundary(canonical)
    provider = OpenAIRealtimeConnection(settings=h.session.settings, owner_id=h.session.context.owner_id)
    provider._websocket = socket
    provider.capture_mode, provider.transport = "auto", "webrtc"
    # Preserve existing fixture helpers, but replace its hand-written writer.
    provider.events = socket.events
    h.provider, h.session.openai = provider, provider
    h.store = canonical
    h.capture.repository = canonical
    h.session._source_context = LiveResponseSources(canonical,
        owner_id=h.session.context.owner_id, session_id=h.session.context.jarvis_session_id,
        audio_session_id=h.session.context.audio_session_id)
    h.session._voice_route_runtime = route
    effects = ControlledTaskEffects(h, canonical)
    for tool in BackgroundTaskToolModule(effects).build_tools(SimpleNamespace(
            owner_id=h.session.context.owner_id, session_id=h.session.context.jarvis_session_id)):
        h.session.tool_registry.register(tool)
    try:
        yield SimpleNamespace(h=h, canonical=canonical, route=route, socket=socket,
                              provider=provider, effects=effects)
    finally:
        # The readiness fixture owns no background resources or runtime shutdown.
        h.session._voice_route_runtime = None
        await h.asyncTearDown()
        assert canonical.depth == 0


async def commit(h, item):
    await h.event("input_audio_buffer.speech_stopped", item_id=item)
    await h.event("input_audio_buffer.committed", item_id=item)


async def admitted_pair(case, *, latest=CURRENT_TEXT):
    h = case.h
    await h.start("old")
    await commit(h, "old")
    await h.until(case.route.entered.is_set)
    await h.transcript("old", OLD_TEXT)
    old_reservation = h.session._pending_response
    old_preparation = h.session._response_preparation_task
    assert old_reservation.submitted is False
    await h.start("new")
    await h.until(old_preparation.done)
    assert old_preparation.cancelled()
    assert case.route.cancelled.is_set()
    assert not h.creates()
    await commit(h, "new")
    await h.transcript("new", latest)
    case.route.release.set()
    await h.until(lambda: len(h.creates()) == 1)
    request = h.creates()[0]
    await h.event("response.created", response={"id": "resp_current", "metadata": request["response"]["metadata"]})
    state = h.session._active_responses["resp_current"]
    offer = state.request_sources
    assert offer is not None
    old = next(item for item in offer.requests if item.kind == "retained_unwritten")
    current = next(item for item in offer.requests if item.kind == "current")
    assert case.canonical.lookups == 0
    assert old.record.id == case.canonical.input("old").id
    assert current.record.id == case.canonical.input("new").id
    assert old_reservation.submitted is False
    return state, old, current


async def dispatch(h, response_id, name, call_id, args):
    item = {"id": call_id + "-item", "type": "function_call", "call_id": call_id,
            "name": name, "arguments": json.dumps(args)}
    await h.event("response.output_item.done", response_id=response_id, item=item)
    return item


def outputs(case):
    return [e["item"] for e in case.socket.events if e["type"] == "conversation.item.create"
            and e.get("item", {}).get("type") == "function_call_output"]


@pytest.mark.asyncio
@pytest.mark.parametrize("include_current_tool", [False, True])
async def test_old_selected_task_source_survives_actual_guarded_followup_wire(include_current_tool):
    async with wire_case() as c:
        h = c.h
        state, old, current = await admitted_pair(c)
        item = await dispatch(h, "resp_current", "cancel_background_task", "cancel-old", {
            "task_ref": c.effects.ref, "expected_revision": 1, SELECTOR: old.reference})
        await h.until(lambda: len(outputs(c)) == 1 and not state.pending_tool_call_ids)
        assert json.loads(outputs(c)[0]["output"])["status"] == "success"
        assert len(c.effects.calls) == len(c.effects.effects) == 1
        selected = c.effects.calls[0][-1]
        assert selected.event_id == old.record.id
        assert selected.revision == old.record.revision
        assert (await h.capture.request_for_call("cancel-old")).id == old.record.id
        items = [item]
        if include_current_tool:
            items.append(await dispatch(h, "resp_current", "fixture_request_evidence", "inspect-current", {
                SELECTOR: current.reference}))
            await h.until(lambda: len(outputs(c)) == 2 and not state.pending_tool_call_ids)
            assert h.invocations[0][0].request_event_id == current.record.id
            assert h.invocations[0][1] == {}
            assert (await h.capture.request_for_call("inspect-current")).id == current.record.id
        # No manually offered continuation: production response.done schedules it.
        await h.event("response.done", response={"id": "resp_current", "status": "completed", "output": items})
        await h.until(lambda: len(h.creates()) == 2)
        followup = h.creates()[1]
        instructions = followup["response"]["instructions"]
        for exact in (OLD_TEXT, CURRENT_TEXT, str(old.record.id), str(current.record.id), "cancel-old"):
            assert exact in instructions
        if include_current_tool:
            assert "inspect-current" in instructions
        assert "tool_request" in instructions and "current" in instructions
        assert followup["response"]["metadata"]["jarvis_turn_generation"] == "2"
        snapshot = h.session._source_context.snapshot_for(followup["event_id"])
        assert {old.record.payload_id, current.record.payload_id}.issubset(snapshot.sources)
        receipt = next(r for r in c.socket.receipts if r["event"] is followup)
        assert receipt["inside_source_transaction"]
        assert {old.record.payload_id, current.record.payload_id, snapshot.payload_id}.issubset(
            receipt["last_source_lock"][1])
        await h.event("response.created", response={"id": "resp_followup", "metadata": followup["response"]["metadata"]})
        follow_state = h.session._active_responses["resp_followup"]
        assert follow_state.request_sources is not None
        assert any(r.record.id == old.record.id for r in follow_state.request_sources.requests)
        # A duplicate old completion cannot dispatch a second continuation/effect.
        await h.event("response.done", response={"id": "resp_current", "status": "completed", "output": items})
        await asyncio.sleep(0)
        assert len(h.creates()) == 2
        assert len(c.effects.calls) == len(c.effects.effects) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("rejection", ["missing", "foreign", "stale"])
async def test_selection_rejected_before_task_effect_at_real_dispatch(rejection):
    async with wire_case() as c:
        h = c.h
        state, old, current = await admitted_pair(c)
        args = {"task_ref": c.effects.ref, "expected_revision": 1}
        if rejection == "foreign":
            args[SELECTOR] = "request_" + uuid4().hex
        elif rejection == "stale":
            args[SELECTOR] = old.reference
            await h.transcript("old", "Correction: do not cancel Workbench Plan.")
        item = await dispatch(h, "resp_current", "cancel_background_task", "denied", args)
        await h.until(lambda: len(outputs(c)) == 1 and not state.pending_tool_call_ids)
        result = json.loads(outputs(c)[0]["output"])
        assert result["status"] == "failed"
        assert result["error"]["code"].startswith("tool_request_source_")
        assert c.effects.calls == []
        assert c.effects.effects == set()
        assert len(h.creates()) == 1


@pytest.mark.asyncio
async def test_newest_correction_is_distinct_in_actual_followup_not_old_cancel_effect():
    async with wire_case() as c:
        h = c.h
        state, old, current = await admitted_pair(c, latest=CORRECTION_TEXT)
        item = await dispatch(h, "resp_current", "fixture_request_evidence", "current-only", {
            SELECTOR: current.reference})
        await h.until(lambda: len(outputs(c)) == 1 and not state.pending_tool_call_ids)
        assert h.invocations[0][0].request_event_id == current.record.id
        assert h.invocations[0][1] == {}
        await h.event("response.done", response={"id": "resp_current", "status": "completed", "output": [item]})
        await h.until(lambda: len(h.creates()) == 2)
        wire = h.creates()[1]
        assert CORRECTION_TEXT in wire["response"]["instructions"]
        assert str(current.record.id) in wire["response"]["instructions"]
        assert c.effects.calls == [] and c.effects.effects == set()
        assert len(h.creates()) == 2


@pytest.mark.asyncio
async def test_unknown_followup_socket_handoff_never_replays_or_reexecutes_tool():
    async with wire_case() as c:
        h = c.h
        state, old, current = await admitted_pair(c)
        item = await dispatch(h, "resp_current", "cancel_background_task", "once", {
            "task_ref": c.effects.ref, "expected_revision": 1, SELECTOR: old.reference})
        await h.until(lambda: len(outputs(c)) == 1 and not state.pending_tool_call_ids)
        assert len(c.effects.effects) == 1
        c.socket.fail_next_create = True
        await h.event("response.done", response={"id": "resp_current", "status": "completed", "output": [item]})
        await h.until(lambda: len(h.creates()) == 2)
        await h.until(lambda: not h.session._provider_completion_tasks)
        assert h.capture._unwritten_inputs == {}
        assert len(c.effects.calls) == len(c.effects.effects) == 1
        assert len(h.creates()) == 2
        # Unknown socket handoff is fenced, not described as unseen/unheard or retried.
        assert h.session._context_revoked or h.session._closed.is_set()
