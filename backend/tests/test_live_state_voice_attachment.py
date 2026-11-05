"""Application attachment and canonical transaction regression tests; no network."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

from starlette.datastructures import QueryParams

from app.errors import JarvisHTTPError
from app.live_state.models import (AggregateRef, FocusEntry, GuardedPayloadBinding,
    LiveSnapshot, SessionScope, SourceDependency, StateCursor)
from app.live_state.service import LiveStateService
from app.realtime.live_state_bridge import SessionLiveStateBridge, install_responsive_voice
from app.realtime.source_context import LiveContextError, LiveResponseSources
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from app.voice_routes.bridge import ContextOfferResult, OfferDisposition
from app.voice_routes.contracts import RouteIdentity, RouteStart, VoiceContextUpdate


NOW = datetime(2026, 10, 30, 20, tzinfo=timezone.utc)
SCOPE = SessionScope(owner_id="fixture-owner", session_id=UUID(int=1), area_id=UUID(int=2))
EPOCH = UUID(int=3)
SOURCE = SourceDependency(source_namespace="context_event", source_id=UUID(int=4), revision=1)
ROOT = UUID(int=5)
BINDING = GuardedPayloadBinding(scope=SCOPE, payload_root_ids=(ROOT,), dependencies=(SOURCE,))


def snapshot(revision=1, *, text="Check the number", expiry=None):
    entry = FocusEntry(entry_id="request", scope=SCOPE, received_at=NOW, observed_at=NOW,
        fresh_until=expiry or NOW + timedelta(seconds=10), sources=(SOURCE,), guard_binding=BINDING,
        evidence_kind="user_reported", goal=text,
        focus=AggregateRef(namespace="context_event", aggregate_id=SOURCE.source_id, revision=1))
    return LiveSnapshot(scope=SCOPE, cursor=StateCursor(epoch=EPOCH, revision=revision),
                        as_of=NOW, readiness="ready", entries=(entry,))


def request(scope=SCOPE):
    route = RouteIdentity(scope=scope, route_id=uuid4(), route_generation=1)
    initial = VoiceContextUpdate(route=route, update_id=uuid4(), cursor=StateCursor(epoch=EPOCH, revision=0),
        mode="snapshot", text="Context availability: warming.", created_at=NOW)
    return RouteStart(route=route, command_id=uuid4(), requested_at=NOW,
                      authorization_ref=scope.session_id, initial_context=initial)


class Connection:
    def __init__(self):
        self.calls = []
        self.active = True

    def is_in_transaction(self):
        return self.active

    async def fetchval(self, statement, owner, roots):
        self.calls.append((owner, tuple(roots)))
        return True


class SourceTransactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_connection_does_not_acquire_pool(self):
        sources = LiveResponseSources(None, owner_id=SCOPE.owner_id, session_id=SCOPE.session_id,
                                      audio_session_id="fixture-audio")
        sources.observe((ROOT,))
        connection = Connection()
        async with sources.context_guard(connection=connection):
            self.assertEqual(connection.calls, [(SCOPE.owner_id, (ROOT,))])

    async def test_external_connection_must_own_transaction(self):
        sources = LiveResponseSources(None, owner_id=SCOPE.owner_id, session_id=SCOPE.session_id,
                                      audio_session_id="fixture-audio")
        sources.observe((ROOT,))
        connection = Connection()
        connection.active = False
        with self.assertRaises(LiveContextError):
            async with sources.context_guard(connection=connection):
                self.fail("An untransactional connection was accepted")

    async def test_runtime_enters_owner_guard_before_root_guard_on_same_connection(self):
        from test_voice_route_runtime import Session, CLIENT_CAPS
        session = Session()
        scope = SessionScope(owner_id=session.context.owner_id, session_id=session.context.jarvis_session_id)
        sources = LiveResponseSources(None, owner_id=scope.owner_id, session_id=scope.session_id,
                                      audio_session_id=session.context.audio_session_id)
        session._source_context = sources
        connection = Connection()
        order = []

        @asynccontextmanager
        async def owner_guard(payload):
            order.append("owner")
            self.assertEqual(connection.calls, [])
            yield connection
            order.append("released")

        start = request(scope)
        runtime = RealtimeVoiceRouteRuntime(session, start, CLIENT_CAPS, atomic_guard=owner_guard,
            payload_roots=lambda _: (ROOT,), validity=AsyncMock(return_value=True),
            authorize_start=AsyncMock(return_value=True), clock=lambda: NOW)
        async with runtime._source_guard(start.initial_context):
            self.assertEqual(order, ["owner"])
            self.assertEqual(connection.calls, [(scope.owner_id, (ROOT,))])
        self.assertEqual(order, ["owner", "released"])


class ProjectionAttachmentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = NOW
        self.current = snapshot()
        self.live = SimpleNamespace(requires_reconstruction=False,
            service=SimpleNamespace(snapshot=lambda: self.current), bind=AsyncMock(return_value=BINDING))
        self.session = SimpleNamespace(_closed=asyncio.Event(), close=AsyncMock())
        self.route = SimpleNamespace(ready=True, request=request(), offer_context=Mock(
            return_value=ContextOfferResult(OfferDisposition.QUEUED)))
        self.bridge = SessionLiveStateBridge(self.session, self.live, self.route, clock=lambda: self.now)

    async def test_background_projection_publishes_actual_bound_text(self):
        await self.bridge.publish_latest()
        update = self.route.offer_context.call_args.args[0]
        self.assertIn("Check the number", update.text)
        self.assertEqual(update.guard_binding, BINDING)
        self.live.bind.assert_awaited_once()

    async def test_unchanged_context_does_not_rebind_or_publish(self):
        await self.bridge.publish_latest()
        self.current = snapshot(2)
        await self.bridge.publish_latest()
        self.live.bind.assert_awaited_once()
        self.route.offer_context.assert_called_once()

    async def test_expiry_renewal_reuses_exact_committed_binding(self):
        await self.bridge.publish_latest()
        self.now += timedelta(seconds=9)
        self.current = snapshot(2, expiry=NOW + timedelta(seconds=20))
        await self.bridge.publish_latest()
        self.live.bind.assert_awaited_once()
        self.assertEqual(self.route.offer_context.call_count, 2)
        self.assertEqual(self.route.offer_context.call_args.args[0].expires_at,
                         NOW + timedelta(seconds=20))

    async def test_newer_selection_during_slow_binding_is_not_published(self):
        async def bind(selection):
            self.current = snapshot(2, text="A different question")
            return BINDING
        self.live.bind.side_effect = bind
        await self.bridge.publish_latest()
        self.route.offer_context.assert_not_called()
        self.assertEqual(self.bridge.last_omission, "selection_changed_during_binding")

    async def test_unready_route_does_not_wait_for_memory(self):
        self.route.ready = False
        await self.bridge.publish_latest()
        self.live.bind.assert_not_awaited()

    async def test_subscription_failure_closes_existing_session(self):
        service = LiveStateService(SCOPE, clock=lambda: NOW)
        await service.start()
        self.live.service = service
        self.live.close = AsyncMock(side_effect=service.close)
        self.route.ready = False
        self.bridge.start()
        await service.close()
        for _ in range(20):
            if self.session.close.await_count:
                break
            await asyncio.sleep(0)
        self.assertEqual(self.bridge.failure, "live_state_subscription_closed")
        await self.bridge.close()
        self.assertTrue(all(task.done() for task in self.bridge._tasks))


class FactoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_factory_drives_existing_native_owner_without_waiting_for_recall(self):
        from test_voice_route_runtime import Session, CLIENT_CAPS, NOW as NATIVE_NOW
        from app.realtime.session import RealtimeAudioSession
        session = Session()
        session.install_voice_route_runtime = lambda runtime: RealtimeAudioSession.install_voice_route_runtime(session, runtime)
        scope = SessionScope(owner_id=session.context.owner_id, session_id=session.context.jarvis_session_id,
                             area_id=SCOPE.area_id)
        service = LiveStateService(scope, clock=lambda: NATIVE_NOW)
        connection = Connection()
        # Reuse the actual accumulated-root guard, not the older fixture guard
        # which predates canonical transaction sharing.
        session._source_context = LiveResponseSources(None, owner_id=scope.owner_id,
            session_id=scope.session_id, audio_session_id=session.context.audio_session_id)

        async def close_session(reason):
            runtime = session._voice_route_runtime
            if runtime is not None:
                await runtime.before_session_close(reason)
            session._closed.set()
            await session.openai.close()
            if runtime is not None:
                await runtime.after_session_close()

        session.close = close_session

        @asynccontextmanager
        async def scope_guard(*, wait_for_owner=True):
            yield connection

        live = SimpleNamespace(service=service, requires_reconstruction=False,
            start=service.start, close=service.close, scope_guard=scope_guard,
            bind=AsyncMock(side_effect=AssertionError("Startup must not wait for recall")))
        areas = SimpleNamespace(read=AsyncMock(return_value={"area_id": str(SCOPE.area_id), "area_status": "active"}))
        services = SimpleNamespace(knowledge_runtime=SimpleNamespace(area_selections=areas))
        with patch("app.realtime.live_state_bridge.create_session_live_state", return_value=live) as create:
            attachment = await install_responsive_voice(session, services,
                SimpleNamespace(RESPONSIVE_VOICE_ENABLED=True), CLIENT_CAPS, clock=lambda: NATIVE_NOW)
        try:
            self.assertEqual(create.call_args.kwargs["scope"], scope)
            live.bind.assert_not_awaited()
            self.assertFalse(attachment.route.ready)
            session._realtime_context_ready.set()
            async with asyncio.timeout(2):
                while not attachment.route.ready:
                    await asyncio.sleep(0)
            self.assertEqual(attachment.route.adapter._context.text, "Context availability: warming.")
            with patch.object(attachment.route, "offer_context",
                              wraps=attachment.route.offer_context) as offer:
                await attachment.publish_latest()
                offer.assert_not_called()
            self.assertIs(attachment._last_update, attachment.route.request.initial_context)
            service.mark_ready()
            async with asyncio.timeout(2):
                while "no current source-backed" not in attachment.route.adapter._context.text:
                    await asyncio.sleep(0)
            live.bind.assert_not_awaited()
            self.assertIs(session._voice_route_runtime, attachment.route)
        finally:
            await attachment.close()
        self.assertTrue(all(task.done() for task in attachment._tasks))
        self.assertTrue(session.openai.closed)
        self.assertTrue(attachment.route.shutdown_confirmed)

    async def test_old_client_does_not_touch_any_new_owner(self):
        result = await install_responsive_voice(object(), object(), object(), None)
        self.assertIsNone(result)

    async def test_explicit_client_does_not_silently_fallback_when_server_disabled(self):
        with self.assertRaises(JarvisHTTPError) as caught:
            await install_responsive_voice(object(), object(), SimpleNamespace(RESPONSIVE_VOICE_ENABLED=False), {})
        self.assertEqual(caught.exception.code, "responsive_voice_disabled")

    async def test_capabilities_must_be_object(self):
        with self.assertRaises(JarvisHTTPError) as caught:
            await install_responsive_voice(object(), object(), SimpleNamespace(RESPONSIVE_VOICE_ENABLED=True), [])
        self.assertEqual(caught.exception.code, "invalid_voice_route_capabilities")

    async def test_missing_canonical_owner_is_explicit(self):
        with self.assertRaises(JarvisHTTPError) as caught:
            await install_responsive_voice(object(), SimpleNamespace(knowledge_runtime=None),
                SimpleNamespace(RESPONSIVE_VOICE_ENABLED=True), {})
        self.assertEqual(caught.exception.code, "live_state_unavailable")


class RouterAttachmentTests(unittest.IsolatedAsyncioTestCase):
    async def run_router(self, capabilities, *, registered=True, registration_error=None):
        from app.realtime.provider_selection import AudioProvider
        from app.realtime.router import audio_realtime_websocket
        from app.runtime import JarvisRuntimeServices
        manager = SimpleNamespace(register=AsyncMock(return_value=registered), unregister=AsyncMock())
        services = Mock(spec=JarvisRuntimeServices)
        services.realtime_session_manager = manager
        services.visual_memory_services = None
        services.tool_modules = ()
        services.assistant_context_providers = ()
        services.sensory_frame_window = None
        services.experience_event_service = None
        session = SimpleNamespace(run=AsyncMock(), close=AsyncMock())
        def activate():
            self.assertEqual(manager.register.await_count, 1)
            self.assertTrue(registered)
            session.run.assert_not_awaited()
            if registration_error is not None:
                raise registration_error

        attachment = SimpleNamespace(close=AsyncMock(), registered=Mock(side_effect=activate))
        offer = {"type": "audio.webrtc.offer", "sdp": "fixture", "capture_mode": "auto"}
        if capabilities is not None:
            offer["voice_route"] = capabilities
        import json
        websocket = SimpleNamespace(accept=AsyncMock(), headers={},
            query_params=QueryParams(f"session_id={SCOPE.session_id}&client_kind=phone_audio&audio_transport=webrtc"),
            app=SimpleNamespace(state=SimpleNamespace(runtime_services=services)),
            receive_text=AsyncMock(return_value=json.dumps(offer)))
        repository = SimpleNamespace(get_session=AsyncMock(return_value=SimpleNamespace(
            id=SCOPE.session_id, status="active")))
        prefix = "app.realtime.router."
        with patch(prefix + "get_settings", return_value=object()), \
             patch(prefix + "authenticate_static_bearer", return_value=SimpleNamespace(owner_id=SCOPE.owner_id)), \
             patch(prefix + "select_audio_provider", return_value=SimpleNamespace(provider=AudioProvider.OPENAI, explicit=False)), \
             patch(prefix + "_require_realtime_configured"), patch(prefix + "get_pool", return_value=object()), \
             patch(prefix + "SessionRepository", return_value=repository), \
             patch(prefix + "_create_context_capture", return_value=None), \
             patch(prefix + "_send_ws_error_and_close", new_callable=AsyncMock) as error, \
             patch("app.realtime.webrtc.validate_offer", return_value=("fixture", "auto")), \
             patch("app.realtime.direct_session.DirectRealtimeAudioSession", return_value=session), \
             patch("app.realtime.live_state_bridge.install_responsive_voice", new_callable=AsyncMock,
                   return_value=attachment) as install:
            await audio_realtime_websocket(websocket)
        return session, attachment, install, manager, error

    async def test_brief_activation_follows_successful_session_admission(self):
        session, attachment, _, manager, error = await self.run_router({})
        attachment.registered.assert_called_once_with()
        session.run.assert_awaited_once()
        manager.unregister.assert_awaited_once()
        error.assert_not_awaited()

    async def test_duplicate_session_never_activates_perception(self):
        _, attachment, _, _, _ = await self.run_router({}, registered=False)
        attachment.registered.assert_not_called()

    async def test_failed_perception_registration_joins_and_unregisters_session(self):
        session, attachment, _, manager, error = await self.run_router(
            {}, registration_error=RuntimeError("already owned"))
        session.run.assert_not_awaited()
        attachment.close.assert_awaited_once_with("voice_session_finished")
        manager.unregister.assert_awaited_once()
        error.assert_awaited_once()

    async def test_actual_offer_capabilities_install_and_join_attachment(self):
        caps = {"supported_versions": [1]}
        session, attachment, install, manager, error = await self.run_router(caps)
        self.assertEqual(install.await_args.args[-1], caps)
        session.run.assert_awaited_once()
        attachment.close.assert_awaited_once_with("voice_session_finished")
        manager.unregister.assert_awaited_once()
        error.assert_not_awaited()

    async def test_legacy_offer_does_not_install_attachment(self):
        session, attachment, install, manager, error = await self.run_router(None)
        install.assert_not_awaited()
        attachment.close.assert_not_awaited()
        session.run.assert_awaited_once()

    async def test_duplicate_session_registration_joins_new_workers(self):
        session, attachment, install, manager, error = await self.run_router({}, registered=False)
        attachment.close.assert_awaited_once_with("voice_route_registration_rejected")
        session.run.assert_not_awaited()
        manager.unregister.assert_not_awaited()
        error.assert_awaited_once()
