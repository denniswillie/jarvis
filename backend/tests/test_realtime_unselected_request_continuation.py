"""Real admission/guarded wire/registered task tools; controlled storage/effects."""
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

from app.tasks.models import TaskError, TaskStatus
from test_realtime_unwritten_followup_wire import (
    CORRECTION_TEXT, CURRENT_TEXT, OLD_TEXT, SELECTOR, admitted_pair, dispatch, outputs, wire_case,
)


def creates(case):
    return [event for event in case.socket.events if event.get("type") == "response.create"]


def install_status_read(case):
    """Keep the registered get handler; control only its repository result."""
    reads = []

    async def voice_detail(owner_id, task_id):
        assert owner_id == case.h.session.context.owner_id
        assert task_id == case.effects.id
        reads.append(task_id)
        task = SimpleNamespace(
            id=task_id, owner_id=owner_id, session_id=case.h.session.context.jarvis_session_id,
            knowledge_area_id=None, revision=1, status=TaskStatus.RUNNING, title="Fixture task",
            mode="consult", reason=None, pending_status=None, updated_at=datetime.now(UTC),
        )
        return task, {"result": None}

    case.effects.repository.voice_detail = voice_detail
    return reads


async def get_current(case, current):
    await dispatch(case.h, "resp_current", "get_background_task", "call_current_status", {
        "task_ref": case.effects.ref, "expected_revision": 1, SELECTOR: current.reference,
    })
    await case.h.until(lambda: len(outputs(case)) == 1)


async def finish_current(case):
    await case.h.event("response.done", response={"id": "resp_current", "status": "completed", "output": []})


async def admitted_followup(case):
    await finish_current(case)
    await case.h.until(lambda: len(creates(case)) == 2)
    wire = creates(case)[-1]
    await case.h.event("response.created", response={
        "id": "resp_status_followup", "metadata": wire["response"]["metadata"],
    })
    return case.h.session._active_responses["resp_status_followup"], wire


@pytest.mark.asyncio
async def test_current_status_followup_preserves_unselected_old_request_and_can_select_it():
    async with wire_case() as case:
        reads = install_status_read(case)
        state, old, current = await admitted_pair(case)
        await case.h.until(lambda: not case.h.session._context_capture._unwritten_inputs)
        assert not case.h.session._context_capture._unwritten_inputs
        await get_current(case, current)
        assert reads == [case.effects.id]
        assert not case.effects.effects
        followup, wire = await admitted_followup(case)
        offer = followup.request_sources
        assert offer is not None, "current-only get dropped the outstanding unselected user request"
        carried = next(request for request in offer.requests if request.input_id == old.input_id)
        latest = next(request for request in offer.requests if request.input_id == current.input_id)
        assert carried.kind == "retained_unwritten"
        assert carried.record == old.record
        assert carried.reference not in {old.reference, current.reference}
        assert latest.reference not in {old.reference, current.reference}
        assert OLD_TEXT in wire["response"]["instructions"]
        assert CURRENT_TEXT in wire["response"]["instructions"]
        assert followup.tool_followup.call_request_ids == (("call_current_status", current.record.id),)
        assert followup.tool_followup.receipts[0].status == "success"
        assert followup.tool_followup.request.event_id == current.record.id
        assert followup.tool_followup.conversation_request.event_id == current.record.id
        cancel_schema = next(tool for tool in wire["response"]["tools"]
                             if tool["name"] == "cancel_background_task")["parameters"]["properties"]
        assert set(cancel_schema[SELECTOR]["enum"]) == {carried.reference, latest.reference}
        assert cancel_schema["suppress_result_announcement"]["type"] == "boolean"
        assert not case.h.session._context_capture._unwritten_inputs
        with patch.object(case.effects, "mutate", wraps=case.effects.mutate) as mutation:
            await dispatch(case.h, "resp_status_followup", "cancel_background_task", "call_old_cancel", {
                "task_ref": case.effects.ref, "expected_revision": 1, SELECTOR: carried.reference,
            })
            await case.h.until(lambda: len(outputs(case)) == 2)
            assert len(case.effects.effects) == 1
            assert mutation.call_count == 1
            source = mutation.call_args.kwargs["source"]
            assert source.event_id == old.record.id and source.revision == old.record.revision
            resolved = await case.h.session._context_capture.request_for_call("call_old_cancel")
            assert resolved.id == old.record.id
            assert followup.tool_followup.conversation_request.event_id == current.record.id
            assert dict(followup.tool_followup.call_request_ids)["call_old_cancel"] == old.record.id
            await dispatch(case.h, "resp_status_followup", "cancel_background_task", "call_old_cancel", {
                "task_ref": case.effects.ref, "expected_revision": 1, SELECTOR: carried.reference,
            })
            assert mutation.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", ["missing", "old_response", "foreign"])
async def test_followup_requires_its_fresh_reference_without_defaulting_or_effects(selector):
    async with wire_case() as case:
        install_status_read(case)
        state, old, current = await admitted_pair(case)
        await get_current(case, current)
        followup, wire = await admitted_followup(case)
        assert followup.request_sources is not None
        arguments = {"task_ref": case.effects.ref, "expected_revision": 1}
        if selector != "missing":
            arguments[SELECTOR] = old.reference if selector == "old_response" else "request_foreign"
        await dispatch(case.h, "resp_status_followup", "cancel_background_task", "call_rejected", arguments)
        await case.h.until(lambda: len(outputs(case)) == 2)
        assert not case.effects.effects
        assert "call_rejected" not in case.h.session._context_capture._selected_call_sources


@pytest.mark.asyncio
async def test_newest_correction_remains_distinct_and_does_not_automatically_execute_old_request():
    async with wire_case() as case:
        reads = install_status_read(case)
        state, old, current = await admitted_pair(case, latest=CORRECTION_TEXT)
        await get_current(case, current)
        followup, wire = await admitted_followup(case)
        assert followup.request_sources is not None
        assert CORRECTION_TEXT in wire["response"]["instructions"]
        assert "Newer explicit corrections or withdrawals take precedence" in wire["response"]["instructions"]
        latest = next(request for request in followup.request_sources.requests if request.kind == "current")
        await dispatch(case.h, "resp_status_followup", "get_background_task", "call_corrected_current", {
            "task_ref": case.effects.ref, "expected_revision": 1, SELECTOR: latest.reference,
        })
        await case.h.until(lambda: len(outputs(case)) == 2)
        assert len(reads) == 2 and not case.effects.effects
        assert (await case.h.session._context_capture.request_for_call("call_corrected_current")).id == current.record.id


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["owner", "epoch", "provider", "runtime", "generation", "revision"])
async def test_carried_offer_cannot_cross_scope_or_rebind_a_corrected_source(change):
    async with wire_case() as case:
        install_status_read(case)
        state, old, current = await admitted_pair(case)
        await get_current(case, current)
        followup = state.tool_followup
        assert followup.source_offer is state.request_sources
        capture = case.h.session._context_capture
        scope, generation = state.request_sources.scope, state.request_sources.generation
        if change == "owner":
            scope = replace(scope, owner_id="foreign")
        elif change == "epoch":
            scope = replace(scope, epoch=uuid4())
        elif change == "provider":
            scope = replace(scope, provider=object())
        elif change == "runtime":
            scope = replace(scope, runtime=object())
        elif change == "generation":
            generation += 1
            capture._generations[generation] = current.input_id
        else:
            capture._request_records[old.input_id] = old.record.model_copy(update={"revision": old.record.revision + 1})
        with pytest.raises(TaskError) as error:
            capture.request_sources(generation, scope, followup)
        assert error.value.code == "tool_request_source_unavailable"
        assert not case.effects.effects


@pytest.mark.asyncio
async def test_actual_source_guard_blocks_withdrawal_before_followup_write():
    async with wire_case() as case:
        install_status_read(case)
        state, old, current = await admitted_pair(case)
        await get_current(case, current)
        case.canonical.unavailable.add(old.record.payload_id)
        await finish_current(case)
        await case.h.until(lambda: case.h.session._context_revoked)
        assert len(creates(case)) == 1
        assert not case.effects.effects


@pytest.mark.asyncio
async def test_unknown_followup_write_never_requeues_old_request_or_replays_effects():
    async with wire_case() as case:
        reads = install_status_read(case)
        state, old, current = await admitted_pair(case)
        await get_current(case, current)
        case.socket.fail_next_create = True
        await finish_current(case)
        await case.h.until(lambda: case.h.session._context_revoked)
        assert len(creates(case)) == 2
        assert len(reads) == 1 and not case.effects.effects
        assert not case.h.session._context_capture._unwritten_inputs


@pytest.mark.asyncio
async def test_late_finalization_does_not_upgrade_an_unavailable_retained_offer():
    async with wire_case() as case:
        install_status_read(case)
        state, old, current = await admitted_pair(case)
        await get_current(case, current)
        missing = replace(old, record=None)
        prior = replace(state.request_sources, requests=(current, missing))
        followup = replace(state.tool_followup, source_offer=prior)
        capture = case.h.session._context_capture
        offered = capture.request_sources(prior.generation, prior.scope, followup)
        carried = next(request for request in offered.requests if request.input_id == old.input_id)
        assert carried.record is None and carried.presentation()["text_status"] == "unavailable"
        capture.bind_selected_call("call_late", carried)
        with pytest.raises(TaskError):
            await capture.request_for_call("call_late")
        assert not case.effects.effects
