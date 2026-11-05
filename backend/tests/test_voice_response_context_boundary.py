"""Accepted as-of delivery and replacement scheduling, offline transport only."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from app.realtime.direct_session import DirectRealtimeAudioSession
from app.realtime.models import RealtimeResponseState, RealtimeSessionContext
from app.realtime.source_context import LiveContextError
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from app.settings import JarvisSettings
from app.voice_routes.coordinator import CoordinationPhase, VoiceRouteCoordinator
from app.voice_routes.contracts import ContextWriteEvidence
import test_app_output_source_addition_race as output_fixtures
import test_realtime_voice_route as route_fixtures


class VoiceResponseContextBoundaryTests(unittest.IsolatedAsyncioTestCase):
    voice = output_fixtures.AppOutputSourceAdditionRaceTests.voice
    send = output_fixtures.AppOutputSourceAdditionRaceTests.send
    refreshes = output_fixtures.AppOutputSourceAdditionRaceTests.refreshes

    async def until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(0)

    async def route(self, *, notification=False):
        voice, sources, connection, pool, socket, sent = self.voice(native=False)
        voice._active_responses.clear()
        voice._pending_response = None
        voice._direct_output_pending = set()
        runtime = object.__new__(RealtimeVoiceRouteRuntime)
        runtime.session = voice
        runtime.request = route_fixtures.make_start()
        if notification:
            # This composition now checks the actual source-owner/session scope,
            # not only the old synthetic dispatch's unconditional current flag.
            scope = runtime.request.route.scope.model_copy(update={
                "owner_id": sources.owner_id, "session_id": sources.session_id})
            runtime.request = route_fixtures.make_start(
                runtime.request.route.model_copy(update={"scope": scope}))
        runtime.channel = SimpleNamespace(ready=True, media_envelope=lambda payload: dict(payload))
        runtime._installed = True
        runtime._closing = runtime._closed = False
        runtime._response_offers = {}
        runtime._payload_roots = lambda update: (UUID(int=1 if update.cursor.revision == 0 else 2),)

        @asynccontextmanager
        async def atomic(_):
            async with pool.acquire() as owned, owned.transaction():
                yield owned
        runtime._atomic_guard = atomic
        runtime._owner_send_text = voice._app_sender.send_text
        voice._app_sender.send_text = runtime._app_send_text
        control = route_fixtures.FakeControl(runtime.request.route)
        adapter = route_fixtures.make_adapter(control)
        # object.__new__ bypasses the runtime constructor; retain the fixture's
        # existing route authorization callback for scope-only replacement.
        runtime._authorize_start = adapter._bindings.authorize_start
        adapter._bindings = replace(adapter._bindings, source_guard=runtime._source_guard,
            context_replacement_guard=runtime._context_replacement_guard,
            on_source_handoff=runtime._source_handoff)
        owner = VoiceRouteCoordinator(runtime.request.route.scope, validity=AsyncMock(return_value=True),
            clock=lambda: route_fixtures.NOW, preflight_expiry_refresh=True)
        runtime.coordinator = owner
        await owner.start(adapter, runtime.request)
        self.addAsyncCleanup(owner.close)
        return voice, sources, connection, pool, socket, sent, runtime, control, adapter, owner

    async def test_accepted_pin_uses_original_as_of_roots_not_new_observations(self):
        voice, sources, connection, pool, socket, sent = self.voice()
        pin = await sources.prepare("request")
        voice._active_responses["resp_fixture"].source_snapshot = pin
        sources.observe((UUID(int=2), UUID(int=3)))
        self.assertTrue(await self.send(voice))
        self.assertEqual(connection.locks[-1], (*pin.sources, pin.payload_id))
        self.assertIsNotNone(pin.prepared_at.tzinfo)
        self.assertIs(sources.snapshot_for("request"), pin)
        self.assertEqual(self.refreshes(voice), [])
        with self.assertRaises(LiveContextError) as stale:
            async with sources.guard(pin):
                self.fail("A delivery pin authorized a new current-version action")
        self.assertEqual(stale.exception.code, "live_context_changed")
        voice._begin_context_rebuild.assert_not_called()

    async def test_matching_real_registration_pins_only_submitted_request(self):
        for submitted in (True, False):
            with self.subTest(submitted=submitted):
                _, sources, connection, pool, _, _ = self.voice()
                session = DirectRealtimeAudioSession(
                    offer="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
                    websocket=SimpleNamespace(close=AsyncMock(), send_text=AsyncMock()),
                    context=RealtimeSessionContext(audio_session_id="pin-fixture", owner_id="fixture-owner",
                        jarvis_session_id=uuid4(), client_kind="rayban_audio", session=SimpleNamespace()),
                    settings=JarvisSettings(_env_file=None),
                    repository=SimpleNamespace(create_message=AsyncMock(), get_recent_messages=AsyncMock(return_value=[])),
                    tool_modules=(), context_providers=())
                self.addAsyncCleanup(session.openai.close)
                session._source_context = sources
                request = session._reserve_response_locked("normal_user_turn", {})
                pin = await sources.prepare(request.request_id)
                request.submitted = submitted
                session._unacknowledged_response_requests[request.request_id] = request
                await session._register_response({"id":"response", "metadata":{
                    "jarvis_request_id":request.request_id,
                    "jarvis_turn_generation":str(request.turn_generation)}})
                state = session._active_responses["response"]
                self.assertFalse(state.interrupted)
                self.assertIs(state.source_snapshot, pin if submitted else None)
                self.assertEqual(state.metadata["jarvis_request_id"], request.request_id)

    async def test_pinned_native_output_survives_age_but_original_withdrawal_still_retires(self):
        voice, sources, connection, pool, socket, sent, runtime, control, adapter, owner = await self.route()
        pin = await sources.prepare("request")
        state = RealtimeResponseState(response_id="resp_fixture", kind="normal_user_turn", source_snapshot=pin)
        voice._active_responses[state.response_id] = state
        owner._bridge._awaiting_fresh_snapshot = True
        sources.observe((UUID(int=3),))
        self.assertFalse(runtime.ready)
        self.assertFalse(runtime._output_ready(False, "foreign-response"))
        self.assertTrue(await voice._send_app_event("audio.webrtc.output_item", response_id=state.response_id,
                                                   item={"id":"item", "type":"message", "phase":"final_answer"}))
        self.assertEqual(connection.locks[-1], (*pin.sources, pin.payload_id))
        self.assertEqual(len(sent), 1)
        connection.rejected.add(pin.sources[0])
        with self.assertRaises(LiveContextError):
            await voice._send_app_event("audio.webrtc.output_item", response_id=state.response_id,
                                       item={"id":"next", "type":"message", "phase":"final_answer"})
        self.assertEqual(len(sent), 1)
        voice._begin_context_rebuild.assert_called_once()

    async def test_adapter_defers_before_delete_and_keeps_only_newest_complete_snapshot(self):
        voice, sources, connection, pool, socket, sent, runtime, control, adapter, owner = await self.route()
        voice._active_responses["answer"] = object()
        first = route_fixtures.make_context(revision=1, text="obsolete pending")
        owner.offer_context(first)
        await self.until(lambda: owner._bridge.status.retry_update_id == first.update_id)
        self.assertEqual(len(control.writes), 1)
        self.assertEqual(owner.status.phase, CoordinationPhase.ACTIVE)
        self.assertNotIn(UUID(int=2), sources._sources)
        latest = route_fixtures.make_context(revision=2, text="newest complete")
        voice._active_responses.clear()
        # Provider-buffer ownership is still a response boundary, not proof of
        # rendered/heard audio. A completion event alone cannot erase it.
        voice._direct_output_pending.add("answer")
        evidence = ContextWriteEvidence()
        held = await adapter.publish_context_tracked(latest, write_evidence=evidence)
        self.assertEqual(held.reason, "context_response_active")
        self.assertTrue(evidence.proven_unwritten)
        self.assertEqual(len(control.writes), 1)
        voice._direct_output_pending.clear()
        observed_readiness = []
        original_write = control._write

        async def record(event):
            observed_readiness.append(runtime.ready)
            return await original_write(event)
        control._write = record
        owner.offer_context(latest)
        await self.until(lambda: owner._bridge.status.last_receipt is not None)
        self.assertEqual(adapter._context, latest)
        self.assertEqual([row["type"] for row in control.writes],
                         ["conversation.item.create", "conversation.item.delete", "conversation.item.create"])
        self.assertTrue(observed_readiness and not any(observed_readiness))
        self.assertTrue(runtime.ready)
        self.assertIn(UUID(int=2), sources._sources)

    async def test_direct_adapter_final_guard_rejects_candidate_after_owned_delete(self):
        voice, sources, connection, pool, socket, sent, runtime, control, adapter, owner = await self.route()
        connection.rejected.add(UUID(int=2))
        evidence = ContextWriteEvidence()
        # Calling the adapter directly bypasses bridge preflight. Scope-only
        # deletion can complete, but the final candidate guard still rejects.
        receipt = await adapter.publish_context_tracked(route_fixtures.make_context(revision=1),
                                                        write_evidence=evidence)
        self.assertEqual((receipt.status, receipt.reason), ("unknown", "context_write_failed"))
        self.assertTrue(evidence.confirmed_deleted_without_create)
        self.assertFalse(evidence.replacement_create_possible)
        self.assertFalse(evidence.proven_unwritten)
        self.assertEqual([row["type"] for row in control.writes],
                         ["conversation.item.create", "conversation.item.delete"])
        self.assertIsNone(adapter._context_item)
        self.assertNotIn(UUID(int=2), sources._sources)
        self.assertFalse(runtime._context_replacement_active)

    async def test_notification_owner_precedes_pin_on_one_borrowed_transaction(self):
        from contextvars import ContextVar
        from app.attention.quiet import OWNER_LOCK
        from app.realtime.voice_route_runtime import _Dispatch

        for withdrawn in (False, True):
            with self.subTest(withdrawn=withdrawn):
                voice, sources, connection, pool, socket, sent, runtime, control, adapter, owner = await self.route(notification=True)
                pin = await sources.prepare("notification-request")
                offer = route_fixtures.make_offer(runtime.request.route)
                audio_session_id = adapter._bindings.audio_session_id
                voice.context = SimpleNamespace(owner_id=runtime.request.route.scope.owner_id,
                    jarvis_session_id=runtime.request.route.scope.session_id, audio_session_id=audio_session_id)
                voice.openai = runtime._input_provider = control
                voice._voice_route_runtime = runtime
                voice._turn_generation = 1
                runtime._input_audio_session_id = audio_session_id
                runtime.channel.identity = runtime.request.route
                runtime.channel.audio_session_id = audio_session_id
                runtime.adapter = adapter
                runtime._notification_output_context = ContextVar("borrowed_notification_output", default=None)
                metadata = {"jarvis_request_id": "notification-request",
                    "voice_route_request_id": "notification-offer-request",
                    "voice_route_offer_id": str(offer.offer_id),
                    "voice_route_id": str(runtime.request.route.route_id),
                    "voice_route_generation": str(runtime.request.route.route_generation)}
                # Establish the synthetic accepted response only after the
                # controlled transport actually accepts this exact request.
                written = await control.send_event({"type": "response.create", "event_id": "notification-request",
                                                     "response": {"metadata": dict(metadata)}})
                self.assertIs(written, True)
                state = RealtimeResponseState(response_id="notification-response", kind="prospective",
                                               metadata=metadata, source_snapshot=pin)
                voice._active_responses[state.response_id] = state
                voice._attention_delivery = SimpleNamespace(runtime=runtime,
                    owns_offer=lambda value: value == offer and value.route == runtime.request.route)
                runtime._payload_roots = lambda _: (UUID(int=1),)
                order = []

                @asynccontextmanager
                async def owner_guard(payload):
                    self.assertIs(payload, offer)
                    async with pool.acquire() as owned, owned.transaction():
                        await owned.execute(OWNER_LOCK, sources.owner_id)
                        order.append("owner")
                        yield owned
                        order.append("owner_exit")
                runtime._atomic_guard = owner_guard
                source_epoch, speech_epoch = adapter._source_epoch, adapter._speech_epoch
                dispatch = _Dispatch(offer, "notification-offer-request", lambda: owner_guard(offer),
                    current=lambda: runtime.ready, used=True, written=written,
                    response_request_id="notification-request",
                    inflight_current=lambda: (adapter._current(source_epoch)
                        and adapter._speech_epoch == speech_epoch and not adapter._expired(offer)
                        and voice._turn_generation == 1))
                runtime._dispatches = {dispatch.request_id: dispatch}
                runtime._response_offers[state.response_id] = dispatch

                async def locked(roots):
                    self.assertIs(sources._current_connection(), connection)
                    self.assertEqual(order[0], "owner")
                    order.append(tuple(roots))
                connection.on_lock = locked
                if withdrawn:
                    connection.rejected.add(pin.payload_id)
                acquisitions, transactions = pool.entered, connection.entered
                sending = voice._send_app_event("audio.webrtc.output_item", response_id=state.response_id,
                    item={"id":"notification-item", "type":"message", "phase":"final_answer"})
                if withdrawn:
                    with self.assertRaises(LiveContextError):
                        await sending
                    socket.send_text.assert_not_awaited()
                    voice._begin_context_rebuild.assert_called_once()
                else:
                    self.assertTrue(await sending)
                    socket.send_text.assert_awaited_once()
                    voice._begin_context_rebuild.assert_not_called()
                    self.assertEqual(order[-1], "owner_exit")
                self.assertEqual(order[:3], ["owner", (UUID(int=1),), (*pin.sources, pin.payload_id)])
                self.assertEqual(pool.entered - acquisitions, 1)
                self.assertEqual(connection.entered - transactions, 1)
                self.assertIsNone(sources._current_connection())
