"""Real router -> async factory -> Live session, with external boundary fixtures.

No provider, DB or device is contacted. This qualifies application composition,
not audible latency, native media behavior or provider turn-taking.
"""
import asyncio
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from starlette.datastructures import QueryParams

from app.realtime.live_session import create_live_session as actual_factory
from app.realtime.router import audio_realtime_websocket
from app.tasks.context_capture import SessionContextCapture
from test_audio_provider_selection import configured_settings
import test_live_session as session_fixtures
from test_live_voice_adapter import LiveFixture
from test_voice_route_client_channel import capabilities


class LiveRouterCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = session_fixtures.LiveSessionTests()
        await self.fixture.asyncSetUp()
        self.created = []
        self.capture_arguments = []
        self.router_task = None
        self.stack = ExitStack()
        self.settings = configured_settings(RESPONSIVE_VOICE_ENABLED=True, OPENAI_LIVE_ENABLED=True,
                                             OPENAI_REALTIME_MODEL=None)
        context = self.fixture.context
        self.phone = self.fixture.phone
        self.phone.accept = AsyncMock()
        self.phone.headers = {}
        self.phone.app = SimpleNamespace(state=SimpleNamespace(runtime_services=self.fixture.services))
        self.phone.query_params = QueryParams({"session_id": str(context.jarvis_session_id),
            "client_kind": "phone_audio", "audio_provider": "openai",
            "audio_transport": "webrtc", "voice_route": "live"})
        adapter_fixture = LiveFixture()
        manifest = capabilities(adapter_fixture.route.capabilities)
        await adapter_fixture.close()
        self.phone.incoming.put_nowait({"type": "audio.webrtc.offer",
            "sdp": "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n", "capture_mode": "auto",
            "voice_route": manifest})
        repository = SimpleNamespace(get_session=AsyncMock(return_value=SimpleNamespace(
            id=context.jarvis_session_id, status="active")))
        self.stack.enter_context(patch("app.realtime.router.get_settings", return_value=self.settings))
        self.stack.enter_context(patch("app.realtime.router.authenticate_static_bearer",
            return_value=SimpleNamespace(owner_id=context.owner_id)))
        self.stack.enter_context(patch("app.realtime.router.get_pool", return_value=self.fixture.db))
        self.stack.enter_context(patch("app.realtime.router.SessionRepository", return_value=repository))
        # Only the services container boundary is adapted; actual manager,
        # session, factory, channel, coordinator, state and binding remain real.
        self.stack.enter_context(patch("app.realtime.router.JarvisRuntimeServices", SimpleNamespace))
        self.stack.enter_context(patch("app.realtime.router._create_context_capture", side_effect=self.capture))
        self.factory = self.stack.enter_context(patch("app.realtime.live_session.create_live_session",
            side_effect=self.make_session))
        self.realtime_factory = self.stack.enter_context(patch(
            "app.realtime.direct_session.DirectRealtimeAudioSession",
            side_effect=AssertionError("A Live request must not start Realtime")))
        self.legacy_attachment = self.stack.enter_context(patch(
            "app.realtime.live_state_bridge.install_responsive_voice", new_callable=AsyncMock,
            side_effect=AssertionError("A Live request must not install a second attachment")))

    def capture(self, services, **values):
        self.assertIs(services, self.fixture.services)
        self.capture_arguments.append(values)
        return SessionContextCapture(self.fixture.context_repository, **values)

    async def make_session(self, **values):
        session = await actual_factory(**values)
        self.created.append(session)
        self.assertIsNone(session.live)
        self.assertFalse(session._tasks)
        return session

    async def until(self, condition):
        async with asyncio.timeout(3):
            while not condition():
                if self.router_task is not None and self.router_task.done():
                    await self.router_task
                    self.fail("Router exited before the required lifecycle boundary")
                await asyncio.sleep(.002)

    async def asyncTearDown(self):
        try:
            for session in self.created:
                await session.close("router_fixture_cleanup")
            if self.router_task is not None:
                await asyncio.gather(self.router_task, return_exceptions=True)
        finally:
            self.stack.close()
            await self.fixture.asyncTearDown()

    async def test_real_router_awaits_factory_and_reaches_gated_live_ready(self):
        self.router_task = asyncio.create_task(audio_realtime_websocket(self.phone))
        await self.until(lambda: bool(self.created) and self.created[0].ready)
        session = self.created[0]
        self.factory.assert_awaited_once()
        self.realtime_factory.assert_not_called()
        self.legacy_attachment.assert_not_awaited()
        self.assertEqual(len(self.fixture.wire.requests), 1)
        self.assertEqual(self.phone.maximum_readers, 1)
        self.assertEqual(self.capture_arguments[0]["provider"], "openai")
        self.assertEqual(session.capture.capture.audio_session_id, session.context.audio_session_id)
        self.assertEqual(session.capture.capture.provider, "openai")
        selected = next(message for message in self.phone.messages if message["type"] == "audio.provider.selected")
        self.assertEqual(selected["model"], "gpt-live-1")
        self.assertEqual([message["type"] for message in self.phone.messages
            if message["type"].startswith("live.media.")], ["live.media.prepare", "live.media.activate"])
        await session.close("router_fixture_finished")
        await self.router_task
        self.assertTrue(session.shutdown_confirmed, session.failure)
        status = await self.fixture.manager.get_session_diagnostics(self.fixture.context.jarvis_session_id)
        self.assertFalse(status["realtime_connected"])

    async def test_duplicate_registration_closes_real_inert_session_without_paid_post(self):
        context = self.fixture.context
        existing = SimpleNamespace(context=context)
        self.assertTrue(await self.fixture.manager.register(audio_session_id=context.audio_session_id,
            owner_id=context.owner_id, jarvis_session_id=context.jarvis_session_id, session=existing))
        try:
            await audio_realtime_websocket(self.phone)
            self.factory.assert_awaited_once()
            self.assertEqual(len(self.created), 1)
            self.assertIsNone(self.created[0].live)
            self.assertTrue(self.created[0].shutdown_confirmed)
            self.assertFalse(self.fixture.wire.requests)
            self.realtime_factory.assert_not_called()
            status = await self.fixture.manager.get_session_diagnostics(context.jarvis_session_id)
            self.assertEqual(status["audio_session_id"], context.audio_session_id)
        finally:
            await self.fixture.manager.unregister(audio_session_id=context.audio_session_id,
                owner_id=context.owner_id, jarvis_session_id=context.jarvis_session_id)
