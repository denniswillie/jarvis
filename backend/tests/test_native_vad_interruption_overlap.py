"""Offline native VAD ownership regression and legacy-cancel counterexample.

The real DirectRealtimeAudioSession owns input admission and response writes.
The real RealtimeVoiceRouteRuntime cancel delegate invokes its captured legacy
client handler. Existing fixtures replace transport and persistence only.

No Swift callback, installed route handshake, provider, SQL or playback runs.
The sole-owner sequence models *absence* of a redundant backend cancellation;
it does not simulate or claim to qualify the client's local output fence.
The counterexample deliberately asserts today's unsafe canceled-commit outcome,
not a proposed change to backend explicit-cancel semantics. Counts are distinct
provider input items, not command retries or measurements of acoustic cuts.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest

from app.realtime.direct_session import DirectRealtimeAudioSession
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
import test_realtime_auto_vad_request_evidence as vad_fixtures
import test_voice_route_runtime as route_fixtures


@asynccontextmanager
async def overlap_fixture():
    # Compose the existing fixture without collecting its tests a second time.
    fixture = vad_fixtures.NativeAutoVADRequestEvidenceTests()
    await fixture.asyncSetUp()
    runtime = None
    try:
        session = fixture.session
        assert isinstance(session, DirectRealtimeAudioSession)
        request = route_fixtures.request_for(session)
        route = request.route.model_copy(update={
            "scope": request.route.scope.model_copy(update={
                "owner_id": session.context.owner_id,
            }),
        })
        request = request.model_copy(update={
            "route": route,
            "initial_context": request.initial_context.model_copy(update={"route": route}),
        })

        @asynccontextmanager
        async def atomic(_):
            yield

        async def valid(_):
            return True

        runtime = RealtimeVoiceRouteRuntime(
            session, request, route_fixtures.CLIENT_CAPS,
            atomic_guard=atomic, payload_roots=lambda _: (), validity=valid,
            authorize_start=valid, clock=lambda: route_fixtures.NOW,
            ready_timeout=0.2, control_timeout=0.05,
        )
        # Only the actual delegation seam is exercised. Do not manufacture a
        # ready/installed route or replace the session's real admission handler.
        assert runtime._owner_client_handler.__self__ is session
        yield fixture, runtime
    finally:
        try:
            if runtime is not None:
                assert await runtime.after_session_close()
        finally:
            await fixture.asyncTearDown()


async def join_requests(fixture):
    tasks = tuple(fixture.session._direct_tasks)
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
    assert not fixture.session._closed.is_set()


async def commit_response(fixture, item_id, response_id):
    session = fixture.session
    generation = session._direct_inputs[item_id]
    before = len(fixture.creates())
    await fixture.event("input_audio_buffer.speech_stopped", item_id=item_id)
    await fixture.event("input_audio_buffer.committed", item_id=item_id)
    await join_requests(fixture)
    assert len(fixture.creates()) == before + 1
    request = fixture.creates()[-1]
    pending = session._unacknowledged_response_requests[request["event_id"]]
    assert pending.submitted
    assert pending.turn_generation == generation == session._turn_generation
    assert item_id in session._direct_committed
    assert not session._user_audio_pending
    await fixture.event("response.created", response={
        "id": response_id, "metadata": request["response"]["metadata"],
    })
    assert not session._active_responses[response_id].interrupted
    assert session._current_response_id == response_id
    # This is a controlled provider-buffer event, not decoded/heard audio.
    await fixture.event("output_audio_buffer.started", response_id=response_id)
    assert response_id in session._direct_audio_responses
    return request


async def rejected_commit(fixture, item_id):
    before = len(fixture.creates())
    await fixture.event("input_audio_buffer.speech_stopped", item_id=item_id)
    await fixture.event("input_audio_buffer.committed", item_id=item_id)
    await join_requests(fixture)
    assert len(fixture.creates()) == before
    assert item_id not in fixture.session._direct_committed


async def seed_output(fixture, enabled):
    if not enabled:
        return None
    await fixture.start("input-seed")
    await commit_response(fixture, "input-seed", "resp_seed")
    return "resp_seed"


@pytest.mark.parametrize("input_count", [1, 2, 3])
@pytest.mark.parametrize("initial_output", [False, True], ids=["idle", "active-output"])
def test_native_vad_sole_owner_preserves_each_new_input(input_count, initial_output):
    async def scenario():
        async with overlap_fixture() as (fixture, _runtime):
            session = fixture.session
            previous_response = await seed_output(fixture, initial_output)
            expected_creates = int(initial_output)
            items = [f"input-sole-owner-{index}" for index in range(input_count)]
            for index, item_id in enumerate(items):
                generation = session._turn_generation + 1
                await fixture.start(item_id)
                assert session._direct_inputs[item_id] == generation
                assert session._turn_generation == generation
                assert session._user_audio_pending and session._normal_turn_pending
                if previous_response is not None:
                    assert session._active_responses[previous_response].interrupted
                    # A genuine old-response terminal callback cannot settle the
                    # newly admitted input. No duplicate runtime cancel is sent.
                    await fixture.event("response.done", response={
                        "id": previous_response, "status": "cancelled", "output": [],
                    })
                    await fixture.event("output_audio_buffer.stopped", response_id=previous_response)
                    assert session._turn_generation == generation
                    assert session._user_audio_pending and session._normal_turn_pending
                previous_response = f"resp_sole_owner_{index}"
                await commit_response(fixture, item_id, previous_response)
                expected_creates += 1
                # Retry behavior is checked separately from the independent
                # item count and must not create a second turn or response.
                await fixture.start(item_id)
                await fixture.event("input_audio_buffer.committed", item_id=item_id)
                await join_requests(fixture)
                assert session._turn_generation == generation
                assert len(fixture.creates()) == expected_creates
            assert set(items) <= session._direct_committed
            assert len({session._direct_inputs[item] for item in items}) == input_count
            assert len({request["event_id"] for request in fixture.creates()}) == expected_creates
            fixture.assert_no_explicit_commit()

    asyncio.run(scenario())


@pytest.mark.parametrize("input_count", [1, 2, 3])
@pytest.mark.parametrize("initial_output", [False, True], ids=["idle", "active-output"])
def test_duplicate_legacy_cancel_reproduces_canceled_commit_counterexample(input_count, initial_output):
    async def scenario():
        async with overlap_fixture() as (fixture, runtime):
            session = fixture.session
            await seed_output(fixture, initial_output)
            items = [f"input-counterexample-{index}" for index in range(input_count)]
            for item_id in items:
                await fixture.start(item_id)
                generation = session._direct_inputs[item_id]
                assert session._user_audio_pending and session._normal_turn_pending
                await runtime._cancel_response()
                assert session._turn_generation == generation + 1
                assert not session._user_audio_pending
                assert not session._normal_turn_pending
                assert not session._normal_response_requested
                assert session._pending_response is None
                assert session._last_user_turn_settled_at is not None
                # The stale item mapping remains observable, but its commit
                # cannot submit a response after legacy cancellation settles it.
                assert session._direct_inputs[item_id] == generation
                await rejected_commit(fixture, item_id)
            assert len(fixture.creates()) == int(initial_output)
            assert len({session._direct_inputs[item] for item in items}) == input_count
            fixture.assert_no_explicit_commit()

    asyncio.run(scenario())


@pytest.mark.parametrize("input_count", [1, 2, 3])
def test_true_subsequent_speech_supersedes_only_older_inputs(input_count):
    async def scenario():
        async with overlap_fixture() as (fixture, _runtime):
            session = fixture.session
            items = [f"input-supersession-{index}" for index in range(input_count + 1)]
            for item_id in items:
                before = session._turn_generation
                await fixture.start(item_id)
                assert session._turn_generation == before + 1
            current = items[-1]
            generation = session._direct_inputs[current]
            for stale in items[:-1]:
                await rejected_commit(fixture, stale)
                assert session._turn_generation == generation
                assert session._user_audio_pending and session._normal_turn_pending
            await commit_response(fixture, current, "resp_supersession")
            for stale in items[:-1]:
                await rejected_commit(fixture, stale)
                assert session._current_response_id == "resp_supersession"
                assert not session._active_responses["resp_supersession"].interrupted
            assert len(fixture.creates()) == 1
            assert session._direct_committed == {current}
            fixture.assert_no_explicit_commit()

    asyncio.run(scenario())


@pytest.mark.parametrize("input_count", [1, 2, 3])
def test_explicit_cancel_still_settles_input_and_allows_a_later_new_turn(input_count):
    async def scenario():
        async with overlap_fixture() as (fixture, _runtime):
            session = fixture.session
            items = [f"input-explicit-cancel-{index}" for index in range(input_count)]
            for index, item_id in enumerate(items):
                await fixture.start(item_id)
                generation = session._direct_inputs[item_id]
                # Distinct explicit controls, not repeated use of one command ID.
                await session._handle_client_event({
                    "type": "response.cancel", "client_event_id": f"explicit-stop-{index}",
                })
                assert session._turn_generation == generation + 1
                assert not session._user_audio_pending
                assert not session._normal_turn_pending
                await rejected_commit(fixture, item_id)
            assert fixture.creates() == []
            await fixture.start("input-after-explicit-cancel")
            await commit_response(fixture, "input-after-explicit-cancel", "resp_after_explicit_cancel")
            assert len(fixture.creates()) == 1
            for stale in items:
                await rejected_commit(fixture, stale)
            assert session._current_response_id == "resp_after_explicit_cancel"
            assert not session._active_responses["resp_after_explicit_cancel"].interrupted
            fixture.assert_no_explicit_commit()

    asyncio.run(scenario())
