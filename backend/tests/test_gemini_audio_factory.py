from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from starlette.datastructures import QueryParams

from app.realtime.provider_selection import AudioProvider, AudioProviderSelection
from app.realtime.router import audio_realtime_websocket
from app.realtime.session_manager import RealtimeSessionManager
from app.runtime import JarvisRuntimeServices
from app.settings import JarvisSettings


class GeminiAudioFactoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_construction_needs_no_openai_credentials_and_releases_registration(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = JarvisSettings(
                _env_file=None, JARVIS_OWNER_ID="fixture-owner",
                JARVIS_STATIC_BEARER_TOKEN="fixture-token", GEMINI_API_KEY="fixture-gemini",
                OPENAI_API_KEY=None, OPENAI_REALTIME_MODEL=None,
            )
        session_id = uuid4()
        manager = RealtimeSessionManager()
        runtime = Mock(spec=JarvisRuntimeServices)
        runtime.realtime_session_manager = manager
        runtime.visual_memory_services = None
        runtime.tool_modules = ()
        runtime.assistant_context_providers = ()
        runtime.sensory_frame_window = None
        runtime.experience_event_service = None
        websocket = SimpleNamespace(
            accept=AsyncMock(), send_json=AsyncMock(), close=AsyncMock(),
            headers={"authorization": "Bearer fixture-token"},
            query_params=QueryParams(f"session_id={session_id}&audio_provider=gemini"),
            app=SimpleNamespace(state=SimpleNamespace(runtime_services=runtime)),
        )
        repository = SimpleNamespace(get_session=AsyncMock(return_value=SimpleNamespace(id=session_id, status="active")))
        selected = AudioProviderSelection(AudioProvider.GEMINI, settings.GEMINI_LIVE_MODEL, True)
        session = SimpleNamespace(run=AsyncMock(), close=AsyncMock())
        # Deliberately override admission for a factory-only test. The normal
        # selector continues to reject Gemini until the remaining stages pass.
        with patch("app.realtime.router.get_settings", return_value=settings), \
                patch("app.realtime.router.select_audio_provider", return_value=selected), \
                patch("app.realtime.router.get_pool", return_value=object()), \
                patch("app.realtime.router.SessionRepository", return_value=repository), \
                patch("app.realtime.router.RealtimeAudioSession", side_effect=AssertionError("GPT must not be constructed")), \
                patch("app.realtime.gemini.session.GeminiAudioSession", return_value=session) as factory:
            await audio_realtime_websocket(websocket)
        session.run.assert_awaited_once()
        self.assertIs(factory.call_args.kwargs["repository"], repository)
        self.assertIs(factory.call_args.kwargs["settings"], settings)
        self.assertEqual(websocket.send_json.call_args.args[0]["provider"], "gemini")
        self.assertFalse((await manager.get_session_diagnostics(session_id))["realtime_connected"])
