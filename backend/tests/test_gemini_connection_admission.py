from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from app.realtime.gemini.client import GeminiTransportError
from app.realtime.gemini.config import GeminiLiveConfig
from app.realtime.gemini.session import GeminiAudioSession
from app.realtime.models import RealtimeSessionContext
from app.realtime.provider_selection import require_gemini_live_enabled
from app.settings import JarvisSettings
from evals.gemini_live.session_probe import ProbePhone


class GeminiConnectionAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def session(self, *, enabled=True, config=None):
        settings = JarvisSettings(_env_file=None, GEMINI_API_KEY="fixture-key", GEMINI_LIVE_ENABLED=enabled)
        connection = SimpleNamespace(connect=AsyncMock(), close=AsyncMock())
        session = GeminiAudioSession(
            websocket=ProbePhone(), context=RealtimeSessionContext(
                audio_session_id=str(uuid4()), owner_id="fixture-owner", jarvis_session_id=uuid4(),
                client_kind="phone_audio", session=SimpleNamespace(behavior_instructions="")),
            settings=settings, repository=SimpleNamespace(), config=config,
            connection_factory=lambda **kwargs: connection,
            connection_admission=lambda: require_gemini_live_enabled(settings),
        )
        session._connection_setup = AsyncMock(return_value={"instructions": "frozen", "tools": []})
        self.addAsyncCleanup(session.close, "test_finished")
        return session, settings, connection

    async def test_disabled_initial_connection_does_not_read_context_or_connect(self):
        session, _, connection = self.session(enabled=False)
        with self.assertRaises(GeminiTransportError) as raised:
            await session._connect_current()
        self.assertEqual(raised.exception.code, "audio_provider_unavailable")
        session._connection_setup.assert_not_awaited()
        connection.connect.assert_not_awaited()

    async def test_enablement_is_rechecked_after_async_context_preparation(self):
        session, settings, connection = self.session()

        async def prepare():
            settings.GEMINI_LIVE_ENABLED = False
            return {"instructions": "frozen", "tools": []}

        session._connection_setup.side_effect = prepare
        with self.assertRaises(GeminiTransportError):
            await session._connect_current()
        connection.connect.assert_not_awaited()

    async def test_enabled_connection_uses_the_selected_frozen_configuration(self):
        config = GeminiLiveConfig(api_key="selected-key", model="selected-model", voice="selected-voice")
        session, settings, connection = self.session(config=config)
        settings.GEMINI_LIVE_MODEL = "later-model"
        await session._connect_current()
        self.assertIs(session.config, config)
        connection.connect.assert_awaited_once_with(instructions="frozen", tools=[])

    async def test_disabled_resumption_cannot_construct_a_new_connection(self):
        session, _, connection = self.session(enabled=False)
        session._connection_factory = Mock()
        with self.assertRaises(GeminiTransportError):
            await session._resume_connection(reason="goaway")
        session._connection_factory.assert_not_called()
        connection.connect.assert_not_awaited()
        connection.close.assert_not_awaited()

    async def test_enablement_change_blocks_a_second_resume_attempt(self):
        session, settings, _ = self.session()
        session._checkpoint_safe = lambda: True
        session._native_setup = {"instructions": "frozen", "tools": []}
        session.recovery = SimpleNamespace(
            handle="private-handle", deadline=None, attempts=0, clear=Mock(),
        )

        async def failed_connect(**kwargs):
            settings.GEMINI_LIVE_ENABLED = False
            raise GeminiTransportError("gemini_connect_failed")

        retry = SimpleNamespace(connect=AsyncMock(side_effect=failed_connect), close=AsyncMock())
        session._connection_factory = Mock(return_value=retry)
        with self.assertRaises(GeminiTransportError):
            await session._resume_connection(reason="disconnect")
        self.assertEqual(session.recovery.attempts, 1)
        session._connection_factory.assert_called_once()
        retry.connect.assert_awaited_once()
