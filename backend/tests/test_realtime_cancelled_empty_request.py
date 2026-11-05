"""Written, then terminally cancelled without output: preserve intent, not effects."""
import asyncio
from dataclasses import replace
import json
from uuid import uuid4

import pytest

from app.assistant.tool_limits import ToolCallBudget
from app.realtime.request_sources import REQUEST_SOURCE_ARGUMENT
from test_realtime_unwritten_request_sources import admitted, call, composed, references


async def written(fx, *, acknowledge=False, final=True):
    fx.route.release()
    await admitted(fx, "old-cancel", "Cancel WorkbenchPlan. Do not announce its result." if final else None)
    await fx.until(lambda: len(fx.creates()) == 1)
    await asyncio.gather(*tuple(fx.session._direct_tasks))
    request = fx.creates()[0]
    if acknowledge:
        await ack(fx, request)
    return request


async def ack(fx, request):
    await fx.event("response.created", response={
        "id": "response-old", "metadata": request["response"]["metadata"]})


async def done(fx, *, status="cancelled", output=()):
    response = {"id": "response-old", "status": status}
    if output is not None:
        response["output"] = list(output)
    await fx.event("response.done", response=response)


async def current(fx):
    await fx.event("input_audio_buffer.committed", item_id="current-question")
    await fx.transcript("current-question", "What is WorkbenchPlan's status? Separately, what is two plus three?")
    await fx.until(lambda: len(fx.creates()) == 2)
    await asyncio.gather(*tuple(fx.session._direct_tasks))
    request = fx.creates()[-1]
    await fx.event("response.created", response={
        "id": "response-current", "metadata": request["response"]["metadata"]})
    return request, fx.session._active_responses["response-current"]


@pytest.mark.parametrize("early_ack", [False, True])
def test_written_cancelled_empty_request_gets_fresh_source_and_exactly_one_effect(early_ack):
    async def run():
        async with composed() as fx:
            old = await written(fx, acknowledge=early_ack)
            await fx.start("current-question")
            if not early_ack:
                await ack(fx, old)
            await done(fx)
            assert fx.capture.diagnostics()["cancelled_empty_requests"] == 1
            assert fx.capture.diagnostics()["unwritten_requests"] == 0
            request, state = await current(fx)
            refs = references(state)
            retained = refs["retained_cancelled_empty"]
            assert "Cancel WorkbenchPlan" in request["response"]["instructions"]
            assert set(state.source_snapshot.sources) >= set(state.request_sources.roots)
            args = {"task_ref": "task_123456abcdef", "expected_revision": 1,
                    REQUEST_SOURCE_ARGUMENT: retained.reference}
            for call_id in ("cancel-one", "cancel-one", "cancel-two"):
                assert (await call(fx, "cancel_background_task", call_id, args))["status"] == "success"
            assert len(fx.tasks.calls) == 2 and len(fx.tasks.effects) == 1
            assert all(source.event_id == retained.record.id for _, _, source in fx.tasks.calls)
            assert state.tool_followup.conversation_request.event_id == refs["current"].record.id
            assert fx.session._turn_generation == 2
            await done(fx)  # Duplicate terminal notification cannot restore consumed context.
            await fx.event("response.output_item.done", response_id="response-old", item={
                "type": "function_call", "id": "late-item", "call_id": "late-call",
                "name": "cancel_background_task", "arguments": json.dumps(args)})
            assert not fx.capture._unwritten_inputs and len(fx.tasks.calls) == 2
    asyncio.run(run())


@pytest.mark.parametrize("case", ["no_terminal", "completed", "failed", "missing_output",
    "terminal_function", "terminal_message", "foreign_epoch", "uncorrelated", "queued_tool", "returned_tool"])
def test_unproved_or_previously_dispatched_response_is_not_retained(case):
    async def run():
        async with composed() as fx:
            old = await written(fx, acknowledge=case in {"queued_tool", "returned_tool"})
            if case in {"queued_tool", "returned_tool"}:
                if case == "queued_tool":
                    fx.session._ordinary_tool_capacity = asyncio.Semaphore(0)
                await fx.event("response.output_item.done", response_id="response-old", item={
                    "type": "function_call", "id": "admitted-item", "call_id": "admitted-call",
                    "name": "fixture_request_evidence", "arguments": "{}"})
                if case == "returned_tool":
                    await fx.until(lambda: not fx.session._ordinary_tool_tasks)
                assert "response-old" in fx.session._responses_with_tool_calls
            await fx.start("current-question")
            if case == "uncorrelated":
                old = {"response": {"metadata": {**old["response"]["metadata"], "jarvis_request_id": "foreign"}}}
            if case not in {"queued_tool", "returned_tool"}:
                await ack(fx, old)
            if case == "foreign_epoch":
                fx.session._visual_context.connection = replace(
                    fx.session._visual_context.connection, connection_epoch=uuid4())
            if case != "no_terminal":
                output = None if case == "missing_output" else ([{"type": "function_call"}]
                    if case == "terminal_function" else [{"type": "message"}]
                    if case == "terminal_message" else [])
                await done(fx, status=case if case in {"completed", "failed"} else "cancelled", output=output)
            if case == "queued_tool":
                fx.session._ordinary_tool_capacity.release()
                await fx.until(lambda: not fx.session._ordinary_tool_tasks)
            assert not fx.capture._unwritten_inputs
            assert not fx.tasks.calls
    asyncio.run(run())


@pytest.mark.parametrize("change", ["revision", "withdrawal", "late_final"])
def test_retained_cancelled_source_still_requires_exact_available_canonical_evidence(change):
    async def run():
        async with composed() as fx:
            old = await written(fx, final=change != "late_final")
            await fx.start("current-question")
            await ack(fx, old)
            await done(fx)
            request, state = await current(fx)
            source = references(state)["retained_cancelled_empty"]
            frozen = json.dumps(request)
            if change == "withdrawal":
                fx.store.withdraw(source.record)
            else:
                await fx.transcript("old-cancel", "Correction: do not cancel WorkbenchPlan.")
            result = await fx.session._select_tool_result(name="cancel_background_task",
                arguments_text=json.dumps({"task_ref": "task_123456abcdef", "expected_revision": 1,
                    REQUEST_SOURCE_ARGUMENT: source.reference}), call_id="invalid-source", source_state=state,
                budget=ToolCallBudget(max_total_calls=20, max_repeated_calls=5))
            assert result["status"] == "failed" and not fx.tasks.calls
            assert json.dumps(request) == frozen
    asyncio.run(run())


def test_offer_prepared_before_terminal_is_unchanged_and_natural_followup_includes_request():
    async def run():
        async with composed() as fx:
            old = await written(fx)
            await fx.start("current-question")
            await fx.event("input_audio_buffer.committed", item_id="current-question")
            await fx.transcript("current-question", "Separately, what is two plus three?")
            await fx.until(lambda: fx.session._pending_response is not None)
            assert len(fx.creates()) == 1
            await ack(fx, old)
            await done(fx)
            await fx.until(lambda: len(fx.creates()) == 2)
            request = fx.creates()[-1]
            assert "retained_cancelled_empty" not in request["response"]["instructions"]
            await fx.event("response.created", response={
                "id": "response-current", "metadata": request["response"]["metadata"]})
            state = fx.session._active_responses["response-current"]
            assert state.request_sources is None
            assert not fx.tasks.calls
            await call(fx, "fixture_request_evidence", "current-lookup", {"value": 5})
            await fx.event("response.done", response={"id": "response-current", "status": "completed", "output": []})
            await fx.until(lambda: len(fx.creates()) == 3)
            assert "retained_cancelled_empty" in fx.creates()[-1]["response"]["instructions"]
            assert "Cancel WorkbenchPlan" in fx.creates()[-1]["response"]["instructions"]
            assert fx.session._turn_generation == 2 and not fx.tasks.calls
    asyncio.run(run())


@pytest.mark.parametrize("cuts", [1, 2, 3])
def test_repeated_empty_cancellations_preserve_all_unfulfilled_sources_with_one_shared_bound(cuts):
    async def run():
        async with composed() as fx:
            request = await written(fx)
            for index in range(cuts):
                item_id, response_id = f"next-{index}", f"interrupted-{index}"
                await fx.start(item_id)
                await fx.event("response.created", response={
                    "id": response_id, "metadata": request["response"]["metadata"]})
                await fx.event("response.done", response={"id": response_id, "status": "cancelled", "output": []})
                await fx.event("input_audio_buffer.committed", item_id=item_id)
                await fx.transcript(item_id, "Separately, what is two plus three? Do not restart the task.")
                await fx.until(lambda: len(fx.creates()) == index + 2)
                await asyncio.gather(*tuple(fx.session._direct_tasks))
                request = fx.creates()[-1]
            await fx.event("response.created", response={
                "id": "response-current", "metadata": request["response"]["metadata"]})
            state = fx.session._active_responses["response-current"]
            assert len(state.request_sources.requests) == cuts + 1
            old = next(r for r in state.request_sources.requests if r.provider_item_id == "old-cancel")
            assert old.kind == "retained_cancelled_empty"
            assert "Cancel WorkbenchPlan" in request["response"]["instructions"]
            result = await call(fx, "cancel_background_task", "final-cancel", {
                "task_ref": "task_123456abcdef", "expected_revision": 1,
                REQUEST_SOURCE_ARGUMENT: old.reference})
            assert result["status"] == "success" and len(fx.tasks.effects) == 1
            assert fx.tasks.calls[0][2].event_id == old.record.id
            assert len(fx.creates()) == cuts + 1
    asyncio.run(run())
