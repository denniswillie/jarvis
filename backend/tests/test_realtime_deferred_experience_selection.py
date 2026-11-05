"""Offline construction checks for the voice-only attachment opt-in.

These checks do not execute tools, enqueue attachments, or imply that an
attachment completed. Execution ordering and service draining belong to the
executor/service tests.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.realtime.direct_session import DirectRealtimeAudioSession
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.settings import JarvisSettings
from app.tools.executor import JarvisToolExecutor


class RealtimeDeferredExperienceSelectionTests(unittest.IsolatedAsyncioTestCase):
    def make_session(self, *, native=False):
        provider = SimpleNamespace(
            capture_mode="auto",
            transport="webrtc" if native else "websocket",
            send_event=AsyncMock(return_value=True),
            close=AsyncMock(),
        )
        repository = SimpleNamespace(
            create_message=AsyncMock(),
            get_recent_messages=AsyncMock(return_value=[]),
        )
        arguments = dict(
            websocket=SimpleNamespace(close=AsyncMock(), send_text=AsyncMock()),
            context=RealtimeSessionContext(
                audio_session_id="deferred-attachment-fixture",
                owner_id="fixture",
                jarvis_session_id=uuid4(),
                client_kind="rayban_audio",
                session=SimpleNamespace(),
            ),
            settings=JarvisSettings(_env_file=None),
            repository=repository,
            tool_modules=(),
            context_providers=(),
        )
        with patch("app.realtime.session.OpenAIRealtimeConnection", return_value=provider), \
                patch("app.realtime.direct_session.OpenAIWebRTCConnection", return_value=provider):
            if native:
                session = DirectRealtimeAudioSession(
                    offer="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n", **arguments)
            else:
                session = RealtimeAudioSession(**arguments)
        self.addAsyncCleanup(session.close, "deferred_attachment_fixture_complete")
        provider.send_event.assert_not_awaited()
        repository.create_message.assert_not_awaited()
        repository.get_recent_messages.assert_not_awaited()
        return session

    def assert_voice_only_opt_in(self, session):
        self.assertTrue(session.tool_executor._defer_experience_attachment)
        self.assertTrue(session.tool_executor._require_source_tracking)
        self.assertFalse(
            session._visual_interrupt_control_executor._defer_experience_attachment)

    async def test_normal_realtime_session_opts_in_only_domain_executor(self):
        self.assert_voice_only_opt_in(self.make_session())

    async def test_native_webrtc_inherits_the_same_opt_in_and_source_guard(self):
        self.assert_voice_only_opt_in(self.make_session(native=True))

    async def test_generic_executor_keeps_synchronous_default(self):
        session = self.make_session()
        generic = JarvisToolExecutor(session.tool_registry)
        self.assertFalse(generic._defer_experience_attachment)
