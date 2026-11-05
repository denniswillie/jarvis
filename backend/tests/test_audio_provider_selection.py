from __future__ import annotations

import asyncio
import builtins
import os
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch, sentinel
from uuid import uuid4

from fastapi import WebSocketDisconnect
from starlette.datastructures import QueryParams

from app.errors import JarvisHTTPError
from app.realtime.provider_selection import AudioProvider, select_audio_provider
from app.realtime.router import audio_realtime_websocket
from app.realtime.session_manager import RealtimeSessionManager
from app.runtime import JarvisRuntimeServices
from app.settings import JarvisSettings


def configured_settings(**overrides) -> JarvisSettings:
    values = {
        "JARVIS_OWNER_ID": "test-owner",
        "JARVIS_STATIC_BEARER_TOKEN": "test-bearer",
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_REALTIME_MODEL": "gpt-realtime-2.1",
    }
    values.update(overrides)
    with patch.dict(os.environ, {}, clear=True):
        return JarvisSettings(_env_file=None, **values)


class AudioProviderSelectionTests(unittest.TestCase):
    def test_defaults_preserve_the_existing_audio_configuration(self):
        settings = configured_settings()
        self.assertEqual(settings.AUDIO_PROVIDER_DEFAULT, "openai")
        self.assertIs(settings.GEMINI_LIVE_ENABLED, False)
        self.assertEqual(settings.GEMINI_LIVE_MODEL, "gemini-3.1-flash-live-preview")
        self.assertEqual(settings.JARVIS_AUDIO_SAMPLE_RATE, 24_000)
        self.assertEqual(settings.JARVIS_AUDIO_FORMAT, "pcm16")
        self.assertEqual(settings.OPENAI_REALTIME_VOICE, "cedar")
        self.assertEqual(settings.GEMINI_VISUAL_MODEL, "gemini-3.1-flash-lite")

    def test_default_selection_is_immutable_and_has_no_explicit_acknowledgement(self):
        settings = configured_settings()
        selected = select_audio_provider(settings=settings, requested_values=[])
        self.assertIs(selected.provider, AudioProvider.OPENAI)
        self.assertFalse(selected.explicit)
        settings.OPENAI_REALTIME_MODEL = "changed-after-selection"
        self.assertEqual(selected.model, "gpt-realtime-2.1")
        with self.assertRaises(FrozenInstanceError):
            selected.model = "another-model"

    def test_explicit_openai_does_not_read_default_or_gemini_configuration(self):
        settings = SimpleNamespace(OPENAI_REALTIME_MODEL="existing-model")
        selected = select_audio_provider(settings=settings, requested_values=["openai"])
        self.assertTrue(selected.explicit)
        self.assertEqual(selected.model, "existing-model")

    def test_selection_event_contains_only_public_routing_metadata(self):
        selected = select_audio_provider(
            settings=configured_settings(), requested_values=["openai"],
        )
        self.assertEqual(selected.as_event(audio_session_id="audio-run"), {
            "type": "audio.provider.selected",
            "protocol_version": 1,
            "audio_session_id": "audio-run",
            "provider": "openai",
            "model": "gpt-realtime-2.1",
        })

    def test_invalid_or_repeated_request_is_not_defaulted(self):
        for values in ([""], ["other"], ["OpenAI"], [" openai "],
                       ["openai", "openai"], ["openai", "gemini"]):
            with self.subTest(values=values), self.assertRaises(JarvisHTTPError) as raised:
                select_audio_provider(settings=configured_settings(), requested_values=values)
            self.assertEqual(raised.exception.status_code, 400)
            self.assertEqual(raised.exception.code, "invalid_audio_provider")

    def test_invalid_default_is_scoped_to_requests_using_it(self):
        settings = configured_settings(AUDIO_PROVIDER_DEFAULT="invalid")
        with self.assertRaises(JarvisHTTPError) as raised:
            select_audio_provider(settings=settings, requested_values=[])
        self.assertEqual(raised.exception.code, "audio_provider_default_invalid")
        self.assertEqual(raised.exception.status_code, 503)
        self.assertIs(
            select_audio_provider(settings=settings, requested_values=["openai"]).provider,
            AudioProvider.OPENAI,
        )

    def test_invalid_unused_gemini_flag_does_not_prevent_settings_or_gpt_selection(self):
        settings = configured_settings(
            GEMINI_LIVE_ENABLED="not-a-boolean", GEMINI_LIVE_MODEL=None,
        )
        for values in ([], ["openai"]):
            with self.subTest(values=values):
                selected = select_audio_provider(settings=settings, requested_values=values)
                self.assertIs(selected.provider, AudioProvider.OPENAI)
        with self.assertRaises(JarvisHTTPError) as raised:
            select_audio_provider(settings=settings, requested_values=["gemini"])
        self.assertEqual(raised.exception.details["reason"], "invalid_enablement")

    def test_gemini_flag_is_parsed_in_its_own_branch_and_never_admits_gpt(self):
        for value, reason in ((False, "disabled"), ("false", "disabled"),
                              ("0", "disabled"), (True, None),
                              ("true", None), ("1", None)):
            settings = configured_settings(
                GEMINI_LIVE_ENABLED=value, GEMINI_API_KEY="test-gemini-key",
                OPENAI_API_KEY=None, OPENAI_REALTIME_MODEL=None,
            )
            for requested in (["gemini"], []):
                settings.AUDIO_PROVIDER_DEFAULT = "gemini"
                if reason is None:
                    selected = select_audio_provider(settings=settings, requested_values=requested)
                    self.assertIs(selected.provider, AudioProvider.GEMINI)
                    self.assertEqual(selected.model, settings.GEMINI_LIVE_MODEL)
                    self.assertEqual(selected.gemini_config.model, selected.model)
                    continue
                with self.subTest(value=value, requested=requested), self.assertRaises(
                    JarvisHTTPError
                ) as raised:
                    select_audio_provider(settings=settings, requested_values=requested)
                self.assertEqual(raised.exception.status_code, 503)
                self.assertEqual(raised.exception.code, "audio_provider_unavailable")
                self.assertEqual(raised.exception.details, {"provider": "gemini", "reason": reason})

    def test_gemini_missing_configuration_is_reported_without_exposing_values(self):
        for key, model in ((None, "gemini-3.1-flash-live-preview"),
                           ("replace_me", "gemini-3.1-flash-live-preview"),
                           ("test-gemini-key", None), ("test-gemini-key", "")):
            settings = configured_settings(
                GEMINI_LIVE_ENABLED=True, GEMINI_API_KEY=key, GEMINI_LIVE_MODEL=model,
            )
            with self.subTest(key=key, model=model), self.assertRaises(JarvisHTTPError) as raised:
                select_audio_provider(settings=settings, requested_values=["gemini"])
            self.assertEqual(raised.exception.code, "gemini_live_not_configured")
            self.assertEqual(raised.exception.details["reason"], "missing_configuration")
            self.assertNotIn("test-gemini-key", str(raised.exception))


class AudioProviderRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.settings = configured_settings()
        self.jarvis_session = SimpleNamespace(id=uuid4(), status="active")
        self.pool = sentinel.pool
        self.repository = SimpleNamespace(
            get_session=AsyncMock(return_value=self.jarvis_session),
        )
        self.manager = RealtimeSessionManager()
        self.runtime = Mock(spec=JarvisRuntimeServices)
        self.runtime.knowledge_runtime = None
        self.runtime.realtime_session_manager = self.manager
        self.runtime.visual_memory_services = None
        self.runtime.tool_modules = (sentinel.tool_module,)
        self.runtime.assistant_context_providers = (sentinel.context_provider,)
        self.runtime.sensory_frame_window = sentinel.sensory_frame_window
        self.runtime.experience_event_service = sentinel.experience_event_service
        self.sessions = []
        self.factory = Mock(side_effect=self.make_session)
        self.pool_getter = Mock(return_value=self.pool)
        self.repository_factory = Mock(return_value=self.repository)
        for name, replacement in {
            "get_settings": Mock(return_value=self.settings),
            "get_pool": self.pool_getter,
            "SessionRepository": self.repository_factory,
            "ReminderRepository": Mock(return_value=sentinel.reminder_repository),
            "ToolExecutionRepository": Mock(return_value=sentinel.tool_execution_repository),
            "RealtimeAudioSession": self.factory,
        }.items():
            self.enterContext(patch(f"app.realtime.router.{name}", replacement))

    def make_session(self, **kwargs):
        async def run():
            await kwargs["websocket"].send_json({
                "type": "session.created",
                "audio_session_id": kwargs["context"].audio_session_id,
            })

        session = SimpleNamespace(run=AsyncMock(side_effect=run), close=AsyncMock())
        self.sessions.append(session)
        return session

    def websocket(self, *providers, authorization="Bearer test-bearer"):
        params = [("session_id", str(self.jarvis_session.id))]
        params.extend(("audio_provider", value) for value in providers)
        return SimpleNamespace(
            accept=AsyncMock(), send_json=AsyncMock(), close=AsyncMock(),
            headers={} if authorization is None else {"authorization": authorization},
            query_params=QueryParams(params),
            app=SimpleNamespace(state=SimpleNamespace(runtime_services=self.runtime)),
        )

    def assert_error(self, websocket, code):
        payload = websocket.send_json.call_args.args[0]
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["code"], code)
        self.assertIs(payload["recoverable"], False)
        websocket.close.assert_awaited_once_with(code=1008, reason=code)
        return payload

    async def assert_unregistered(self):
        self.assertEqual(await self.manager.get_session_diagnostics(self.jarvis_session.id), {
            "realtime_connected": False, "audio_session_id": None,
        })

    async def test_legacy_gpt_keeps_constructor_inputs_and_event_sequence(self):
        websocket = self.websocket()
        await audio_realtime_websocket(websocket)
        self.factory.assert_called_once()
        kwargs = self.factory.call_args.kwargs
        context = kwargs["context"]
        self.assertEqual(kwargs, {
            "websocket": websocket,
            "context": context,
            "settings": self.settings,
            "repository": self.repository,
            "tool_modules": self.runtime.tool_modules,
            "context_providers": self.runtime.assistant_context_providers,
            "visual_repository": None,
            "sensory_frame_window": self.runtime.sensory_frame_window,
            "reminder_repository": sentinel.reminder_repository,
            "tool_execution_repository": sentinel.tool_execution_repository,
            "experience_event_service": self.runtime.experience_event_service,
        })
        self.assertEqual(context.owner_id, "test-owner")
        self.assertEqual(context.jarvis_session_id, self.jarvis_session.id)
        self.assertEqual(context.client_kind, "rayban_audio")
        self.assertIs(context.session, self.jarvis_session)
        self.repository.get_session.assert_awaited_once_with("test-owner", self.jarvis_session.id)
        self.assertEqual([call.args[0]["type"] for call in websocket.send_json.call_args_list],
                         ["session.created"])
        await self.assert_unregistered()

    async def test_explicit_gpt_acknowledgement_precedes_unchanged_session_run(self):
        websocket = self.websocket("openai")
        await audio_realtime_websocket(websocket)
        events = [call.args[0] for call in websocket.send_json.call_args_list]
        self.assertEqual(events[0], {
            "type": "audio.provider.selected", "protocol_version": 1,
            "audio_session_id": self.factory.call_args.kwargs["context"].audio_session_id,
            "provider": "openai", "model": self.settings.OPENAI_REALTIME_MODEL,
        })
        self.assertEqual(events[1]["type"], "session.created")
        self.assertEqual(len(events), 2)
        self.assertNotIn(self.settings.OPENAI_API_KEY, str(events))
        self.sessions[0].run.assert_awaited_once()
        await self.assert_unregistered()

    async def test_gpt_does_not_import_gemini_or_require_its_live_configuration(self):
        self.settings.GEMINI_LIVE_ENABLED = "invalid-unused-flag"
        self.settings.GEMINI_API_KEY = None
        self.settings.GEMINI_LIVE_MODEL = None
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith(("app.realtime.gemini", "evals.gemini_live", "google.genai", "gemini")):
                raise AssertionError("GPT attempted to import a Gemini Live implementation")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            for providers in ((), ("openai",)):
                await audio_realtime_websocket(self.websocket(*providers))
        self.assertEqual(self.factory.call_count, 2)
        await self.assert_unregistered()

    async def test_disabled_gemini_never_constructs_gpt_or_accesses_storage(self):
        for enabled in (False, "false"):
            self.settings.GEMINI_LIVE_ENABLED = enabled
            self.settings.GEMINI_API_KEY = "test-gemini-key"
            with self.subTest(enabled=enabled):
                websocket = self.websocket("gemini")
                await audio_realtime_websocket(websocket)
                self.assert_error(websocket, "audio_provider_unavailable")
        self.factory.assert_not_called()
        self.pool_getter.assert_not_called()
        self.repository_factory.assert_not_called()
        await self.assert_unregistered()

    async def test_authentication_precedes_provider_configuration(self):
        for authorization, code in ((None, "jarvis_auth_required"),
                                     ("Bearer wrong", "jarvis_auth_invalid")):
            with self.subTest(authorization=authorization):
                websocket = self.websocket("gemini", authorization=authorization)
                await audio_realtime_websocket(websocket)
                self.assert_error(websocket, code)
        self.factory.assert_not_called()
        self.pool_getter.assert_not_called()

    async def test_invalid_provider_queries_are_rejected_without_gpt_fallback(self):
        for values in (("",), ("unknown",), ("openai", "openai"), ("openai", "gemini")):
            with self.subTest(values=values):
                websocket = self.websocket(*values)
                await audio_realtime_websocket(websocket)
                self.assert_error(websocket, "invalid_audio_provider")
        self.factory.assert_not_called()
        self.pool_getter.assert_not_called()

    async def test_existing_openai_configuration_error_is_preserved(self):
        for field in ("OPENAI_API_KEY", "OPENAI_REALTIME_MODEL"):
            original = getattr(self.settings, field)
            setattr(self.settings, field, None)
            for providers in ((), ("openai",)):
                with self.subTest(field=field, providers=providers):
                    websocket = self.websocket(*providers)
                    await audio_realtime_websocket(websocket)
                    payload = self.assert_error(websocket, "realtime_not_configured")
                    self.assertEqual(payload["message"], "OpenAI Realtime is not configured.")
            setattr(self.settings, field, original)
        self.factory.assert_not_called()
        self.pool_getter.assert_not_called()

    async def test_existing_session_and_client_kind_validation_is_preserved(self):
        cases = (
            ([], "missing_session_id"),
            ([("session_id", "not-a-uuid")], "invalid_session_id"),
            ([("session_id", str(self.jarvis_session.id)), ("client_kind", "unknown")],
             "unsupported_client_kind"),
        )
        for params, code in cases:
            with self.subTest(code=code):
                websocket = self.websocket()
                websocket.query_params = QueryParams(params)
                await audio_realtime_websocket(websocket)
                self.assert_error(websocket, code)
        self.factory.assert_not_called()
        self.pool_getter.assert_not_called()

    async def test_session_lookup_remains_owner_scoped(self):
        self.repository.get_session.return_value = None
        websocket = self.websocket("openai")
        await audio_realtime_websocket(websocket)
        self.assert_error(websocket, "session_not_found")
        self.repository.get_session.assert_awaited_once_with("test-owner", self.jarvis_session.id)
        self.factory.assert_not_called()

    async def test_closed_session_and_missing_runtime_do_not_announce_selection(self):
        self.jarvis_session.status = "closed"
        websocket = self.websocket("openai")
        await audio_realtime_websocket(websocket)
        self.assert_error(websocket, "session_closed")
        self.jarvis_session.status = "active"
        websocket = self.websocket("openai")
        websocket.app.state.runtime_services = None
        await audio_realtime_websocket(websocket)
        self.assert_error(websocket, "realtime_manager_not_configured")
        self.factory.assert_not_called()

    async def test_duplicate_registration_does_not_replace_existing_session(self):
        await self.manager.register(
            audio_session_id="existing-run", owner_id="test-owner",
            jarvis_session_id=self.jarvis_session.id, session=sentinel.existing_session,
        )
        websocket = self.websocket("openai")
        await audio_realtime_websocket(websocket)
        self.assert_error(websocket, "audio_session_already_active")
        self.sessions[0].run.assert_not_awaited()
        self.assertEqual(await self.manager.get_session_diagnostics(self.jarvis_session.id), {
            "realtime_connected": True, "audio_session_id": "existing-run",
        })
        await self.manager.unregister(
            audio_session_id="existing-run", owner_id="test-owner",
            jarvis_session_id=self.jarvis_session.id,
        )

    async def test_disconnect_during_selection_closes_unstarted_session(self):
        websocket = self.websocket("openai")
        websocket.send_json.side_effect = WebSocketDisconnect(code=1006)
        await audio_realtime_websocket(websocket)
        self.sessions[0].close.assert_awaited_once_with("client_disconnect")
        self.sessions[0].run.assert_not_awaited()
        await self.assert_unregistered()

    async def test_cancellation_during_selection_cleans_up_registration(self):
        websocket = self.websocket("openai")
        websocket.send_json.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await audio_realtime_websocket(websocket)
        self.sessions[0].close.assert_awaited_once_with("provider_selection_failed")
        self.sessions[0].run.assert_not_awaited()
        await self.assert_unregistered()

    async def test_unexpected_selection_write_error_is_not_swallowed(self):
        websocket = self.websocket("openai")
        websocket.send_json.side_effect = RuntimeError("unexpected write failure")
        with self.assertRaisesRegex(RuntimeError, "unexpected write failure"):
            await audio_realtime_websocket(websocket)
        self.sessions[0].close.assert_awaited_once_with("provider_selection_failed")
        self.sessions[0].run.assert_not_awaited()
        await self.assert_unregistered()

    async def test_run_failure_releases_registration_and_allows_next_gpt_start(self):
        def failing_session(**kwargs):
            session = self.make_session(**kwargs)
            session.run.side_effect = JarvisHTTPError(503, "provider_setup_failed", "Setup failed.")
            return session

        self.factory.side_effect = failing_session
        websocket = self.websocket("openai")
        await audio_realtime_websocket(websocket)
        self.assert_error(websocket, "provider_setup_failed")
        await self.assert_unregistered()
        self.factory.side_effect = self.make_session
        await audio_realtime_websocket(self.websocket())
        self.sessions[-1].run.assert_awaited_once()
        await self.assert_unregistered()

    async def test_failed_gemini_request_does_not_prevent_a_gpt_start(self):
        websocket = self.websocket("gemini")
        await audio_realtime_websocket(websocket)
        self.assert_error(websocket, "audio_provider_unavailable")
        await audio_realtime_websocket(self.websocket("openai"))
        self.factory.assert_called_once()
        self.sessions[0].run.assert_awaited_once()
        await self.assert_unregistered()

    def enable_gemini(self):
        self.settings.GEMINI_LIVE_ENABLED = True
        self.settings.GEMINI_API_KEY = "test-gemini-key"

    async def test_enabled_gemini_uses_its_factory_and_acknowledges_before_readiness(self):
        self.enable_gemini()
        self.settings.OPENAI_API_KEY = self.settings.OPENAI_REALTIME_MODEL = None
        websocket = self.websocket("gemini")
        with patch("app.realtime.gemini.session.GeminiAudioSession", side_effect=self.make_session) as gemini:
            await audio_realtime_websocket(websocket)
        self.factory.assert_not_called()
        kwargs = gemini.call_args.kwargs
        self.assertEqual(kwargs["config"].model, self.settings.GEMINI_LIVE_MODEL)
        self.assertIs(kwargs["settings"], self.settings)
        self.assertIs(kwargs["tool_modules"], self.runtime.tool_modules)
        events = [call.args[0] for call in websocket.send_json.call_args_list]
        self.assertEqual([event["type"] for event in events], ["audio.provider.selected", "session.created"])
        self.assertEqual(events[0]["provider"], "gemini")
        self.assertEqual(events[0]["model"], kwargs["config"].model)
        self.assertEqual(events[0]["audio_session_id"], events[1]["audio_session_id"])
        self.assertNotIn("test-gemini-key", str(events))
        kwargs["connection_admission"]()
        self.settings.GEMINI_LIVE_ENABLED = False
        with self.assertRaises(JarvisHTTPError):
            kwargs["connection_admission"]()
        await self.assert_unregistered()

    async def test_gemini_selection_freezes_configuration_before_async_session_lookup(self):
        self.enable_gemini()
        original_model = self.settings.GEMINI_LIVE_MODEL

        async def lookup(*args):
            self.settings.GEMINI_LIVE_MODEL = "changed-after-selection"
            self.settings.GEMINI_LIVE_VOICE = "another-voice"
            return self.jarvis_session

        self.repository.get_session.side_effect = lookup
        websocket = self.websocket("gemini")
        with patch("app.realtime.gemini.session.GeminiAudioSession", side_effect=self.make_session) as gemini:
            await audio_realtime_websocket(websocket)
        self.assertEqual(gemini.call_args.kwargs["config"].model, original_model)
        self.assertEqual(gemini.call_args.kwargs["config"].voice, "Puck")
        self.assertEqual(websocket.send_json.call_args_list[0].args[0]["model"], original_model)
        await self.assert_unregistered()

    async def test_gemini_default_also_sends_provider_acknowledgement(self):
        self.enable_gemini()
        self.settings.AUDIO_PROVIDER_DEFAULT = "gemini"
        websocket = self.websocket()
        with patch("app.realtime.gemini.session.GeminiAudioSession", side_effect=self.make_session):
            await audio_realtime_websocket(websocket)
        self.assertEqual(websocket.send_json.call_args_list[0].args[0]["provider"], "gemini")
        self.factory.assert_not_called()

    async def test_gemini_constructor_failure_is_sanitized_and_gpt_still_starts(self):
        self.enable_gemini()
        websocket = self.websocket("gemini")
        with patch("app.realtime.gemini.session.GeminiAudioSession", side_effect=RuntimeError("private-value")):
            await audio_realtime_websocket(websocket)
        payload = self.assert_error(websocket, "gemini_live_initialization_failed")
        self.assertNotIn("private-value", str(payload))
        self.factory.assert_not_called()
        await self.assert_unregistered()
        await audio_realtime_websocket(self.websocket("openai"))
        self.factory.assert_called_once()
        await self.assert_unregistered()

    async def test_gemini_import_failure_does_not_break_gpt(self):
        self.enable_gemini()
        original_import = builtins.__import__

        def broken_gemini(name, *args, **kwargs):
            if name == "app.realtime.gemini.session":
                raise ImportError("private-dependency-detail")
            return original_import(name, *args, **kwargs)

        websocket = self.websocket("gemini")
        with patch("builtins.__import__", side_effect=broken_gemini):
            await audio_realtime_websocket(websocket)
            await audio_realtime_websocket(self.websocket("openai"))
        self.assert_error(websocket, "gemini_live_initialization_failed")
        self.factory.assert_called_once()
        await self.assert_unregistered()

    async def test_enabled_gemini_retains_owner_and_closed_session_guards(self):
        self.enable_gemini()
        with patch("app.realtime.gemini.session.GeminiAudioSession") as gemini:
            self.repository.get_session.return_value = None
            websocket = self.websocket("gemini")
            await audio_realtime_websocket(websocket)
            self.assert_error(websocket, "session_not_found")
            self.repository.get_session.assert_awaited_once_with("test-owner", self.jarvis_session.id)
            self.repository.get_session.return_value = self.jarvis_session
            self.jarvis_session.status = "closed"
            websocket = self.websocket("gemini")
            await audio_realtime_websocket(websocket)
            self.assert_error(websocket, "session_closed")
        gemini.assert_not_called()
        self.factory.assert_not_called()

    async def test_gemini_ack_disconnect_closes_and_unregisters_the_unstarted_run(self):
        self.enable_gemini()
        websocket = self.websocket("gemini")
        websocket.send_json.side_effect = WebSocketDisconnect(code=1006)
        with patch("app.realtime.gemini.session.GeminiAudioSession", side_effect=self.make_session):
            await audio_realtime_websocket(websocket)
        self.sessions[0].close.assert_awaited_once_with("client_disconnect")
        self.sessions[0].run.assert_not_awaited()
        await self.assert_unregistered()
