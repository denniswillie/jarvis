"""Explicit Live admission and real router dispatch, without provider or DB I/O."""
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from starlette.datastructures import QueryParams
from starlette.websockets import WebSocketState

from app.errors import JarvisHTTPError
from app.realtime.live_selection import select_live_voice_route, require_live_client_offer
from app.realtime.provider_selection import AudioProvider
from app.realtime.router import audio_realtime_websocket
from app.runtime import JarvisRuntimeServices
from test_audio_provider_selection import configured_settings


class LiveRouteSelectionTests(unittest.TestCase):
    def settings(self, **overrides):
        return configured_settings(**(dict(RESPONSIVE_VOICE_ENABLED=True,
            OPENAI_LIVE_ENABLED=True) | overrides))

    def select(self, settings=None, **overrides):
        return select_live_voice_route(**(dict(settings=settings or self.settings(),
            requested_values=["live"], provider=AudioProvider.OPENAI, transport="webrtc") | overrides))

    def test_default_and_explicit_realtime_never_read_live_settings(self):
        for requested in ([], ["realtime"]):
            with self.subTest(requested=requested):
                self.assertIsNone(self.select(SimpleNamespace(), requested_values=requested))
        self.assertIsNone(self.select(SimpleNamespace(), requested_values=[], provider=AudioProvider.GEMINI))

    def test_live_is_disabled_by_default(self):
        with self.assertRaises(JarvisHTTPError) as raised:
            self.select(configured_settings())
        self.assertEqual(raised.exception.code, "live_voice_disabled")

    def test_both_enablement_flags_are_required_and_parsed(self):
        for overrides in ({"RESPONSIVE_VOICE_ENABLED": False}, {"OPENAI_LIVE_ENABLED": False},
                          {"OPENAI_LIVE_ENABLED": "false"}):
            with self.subTest(overrides=overrides), self.assertRaises(JarvisHTTPError):
                self.select(self.settings(**overrides))
        self.assertEqual(self.select(self.settings(OPENAI_LIVE_ENABLED="true")).model, "gpt-live-1")

    def test_invalid_unused_live_flag_does_not_break_legacy_settings(self):
        settings = self.settings(OPENAI_LIVE_ENABLED="invalid")
        self.assertIsNone(self.select(settings, requested_values=[]))
        with self.assertRaises(JarvisHTTPError) as raised:
            self.select(settings)
        self.assertEqual(raised.exception.code, "live_voice_not_configured")

    def test_selection_freezes_model_and_voice_without_needing_realtime_model(self):
        settings = self.settings(OPENAI_REALTIME_MODEL=None)
        chosen = self.select(settings)
        self.assertEqual((chosen.model, chosen.voice), ("gpt-live-1", "marin"))
        settings.OPENAI_LIVE_MODEL = "changed"
        self.assertEqual(chosen.model, "gpt-live-1")
        with self.assertRaises(FrozenInstanceError):
            chosen.voice = "changed"

    def test_invalid_selectors_and_cross_provider_transport_fail_without_fallback(self):
        for requested in ([""], ["Live"], [" live "], ["unknown"], ["live", "live"], ["live", "realtime"]):
            with self.subTest(requested=requested), self.assertRaises(JarvisHTTPError):
                self.select(requested_values=requested)
        for changes in ({"provider": AudioProvider.GEMINI}, {"transport": "websocket"}):
            with self.subTest(changes=changes), self.assertRaises(JarvisHTTPError):
                self.select(**changes)

    def test_incomplete_configuration_is_rejected_without_exposing_values(self):
        for field, value in (("OPENAI_API_KEY", None), ("OPENAI_LIVE_MODEL", "replace_me"),
                             ("OPENAI_LIVE_VOICE", ""), ("OPENAI_LIVE_MODEL", "bad\nmodel")):
            with self.subTest(field=field), self.assertRaises(JarvisHTTPError) as raised:
                self.select(self.settings(**{field: value}))
            self.assertEqual(raised.exception.code, "live_voice_not_configured")

    def test_manual_capture_and_missing_caps_are_not_treated_as_live_support(self):
        for mode, capabilities in (("manual", {"schema_version": 1}), ("auto", None),
                                   ("auto", {}), ("auto", [])):
            with self.subTest(mode=mode, capabilities=capabilities), self.assertRaises(JarvisHTTPError):
                require_live_client_offer(capture_mode=mode, client_capabilities=capabilities)
        require_live_client_offer(capture_mode="auto", client_capabilities={"schema_version": 1})


class LiveRouterDispatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session_id = uuid4()
        self.session = SimpleNamespace(run=AsyncMock(), close=AsyncMock())
        self.manager = SimpleNamespace(register=AsyncMock(return_value=True), unregister=AsyncMock())
        self.services = object.__new__(JarvisRuntimeServices)
        self.services.realtime_session_manager = self.manager
        self.services.visual_memory_services = None
        self.services.knowledge_runtime = None
        self.services.background_task_service = None
        self.services.tool_modules = ()
        self.services.assistant_context_providers = ()
        self.services.sensory_frame_window = None
        self.services.experience_event_service = None
        self.settings = configured_settings(RESPONSIVE_VOICE_ENABLED=True, OPENAI_LIVE_ENABLED=True)
        self.repository = SimpleNamespace(get_session=AsyncMock(return_value=SimpleNamespace(
            id=self.session_id, status="active")))
        self.ws = SimpleNamespace(headers={},
            app=SimpleNamespace(state=SimpleNamespace(runtime_services=self.services)),
            application_state=WebSocketState.CONNECTED, client_state=WebSocketState.CONNECTED,
            accept=AsyncMock(), close=AsyncMock(), send_text=AsyncMock(), send_json=AsyncMock(),
            receive_text=AsyncMock())
        self.capabilities = {"schema_version": 1, "fixture": "actual negotiation is covered by LiveClientChannel tests"}
        self.set_offer()

    def set_offer(self, *, mode="auto", capabilities=True):
        self.offer = {"type": "audio.webrtc.offer", "sdp": "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
                      "capture_mode": mode}
        if capabilities:
            self.offer["voice_route"] = self.capabilities
        self.ws.receive_text.return_value = json.dumps(self.offer)

    async def run_router(self, *, selector="live"):
        query = {"session_id": str(self.session_id), "audio_provider": "openai", "audio_transport": "webrtc"}
        if selector is not None:
            query["voice_route"] = selector
        self.ws.query_params = QueryParams(query)
        with patch("app.realtime.router.get_settings", return_value=self.settings), \
             patch("app.realtime.router.authenticate_static_bearer", return_value=SimpleNamespace(owner_id="test-owner")), \
             patch("app.realtime.router.get_pool", return_value=object()), \
             patch("app.realtime.router.SessionRepository", return_value=self.repository), \
             patch("app.realtime.router._create_live_session", return_value=self.session) as live_factory, \
             patch("app.realtime.direct_session.DirectRealtimeAudioSession", return_value=self.session) as realtime_factory, \
             patch("app.realtime.live_state_bridge.install_responsive_voice", new_callable=AsyncMock) as attach:
            attach.return_value = None
            await audio_realtime_websocket(self.ws)
        return live_factory, realtime_factory, attach

    async def test_explicit_live_dispatches_without_realtime_model_or_attachment(self):
        self.settings.OPENAI_REALTIME_MODEL = None
        live, realtime, attach = await self.run_router()
        live.assert_awaited_once()
        realtime.assert_not_called()
        attach.assert_not_awaited()
        self.assertEqual(live.call_args.kwargs["route_configuration"].model, "gpt-live-1")
        self.assertEqual(live.call_args.kwargs["client_capabilities"], self.capabilities)
        self.assertIs(self.manager.register.call_args.kwargs["session"], self.session)
        self.session.run.assert_awaited_once()
        self.session.close.assert_awaited_once_with("voice_session_finished")
        self.manager.unregister.assert_awaited_once()

    async def test_no_live_selector_preserves_realtime_factory(self):
        live, realtime, attach = await self.run_router(selector=None)
        live.assert_not_called()
        realtime.assert_called_once()
        attach.assert_awaited_once()
        self.session.run.assert_awaited_once()

    async def test_disabled_live_never_constructs_or_falls_back(self):
        self.settings.OPENAI_LIVE_ENABLED = False
        live, realtime, attach = await self.run_router()
        live.assert_not_called()
        realtime.assert_not_called()
        attach.assert_not_awaited()
        self.manager.register.assert_not_awaited()

    async def test_invalid_native_offer_never_constructs_live(self):
        for mode, capabilities in (("manual", True), ("auto", False)):
            with self.subTest(mode=mode, capabilities=capabilities):
                self.set_offer(mode=mode, capabilities=capabilities)
                live, realtime, attach = await self.run_router()
                live.assert_not_called()
                realtime.assert_not_called()
                attach.assert_not_awaited()

    async def test_rejected_manager_admission_closes_without_starting_provider(self):
        self.manager.register.return_value = False
        await self.run_router()
        self.session.run.assert_not_awaited()
        self.session.close.assert_awaited_once_with("voice_route_registration_rejected")
        self.manager.unregister.assert_not_awaited()

    async def test_run_failure_still_closes_and_unregisters(self):
        self.session.run.side_effect = RuntimeError("controlled session failure")
        with self.assertRaisesRegex(RuntimeError, "controlled session failure"):
            await self.run_router()
        self.session.close.assert_awaited_once_with("voice_session_finished")
        self.manager.unregister.assert_awaited_once()
