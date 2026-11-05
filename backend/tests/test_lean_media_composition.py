"""Real Lean constructor advertises only an actually composed media adapter."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.realtime.lean_session import LeanRealtimeAudioSession
from app.realtime.models import RealtimeSessionContext
from app.runtime.feature_config import TaskLimits
from app.settings import JarvisSettings
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from test_lean_voice_session import FakeProvider, FakeSocket


class LeanMediaCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def make(self, *, configured=True, files_present=True, recording_present=True, wrong_pool=False,
                   journal_present=True):
        pool = object()
        sources = SimpleNamespace(pool=pool)
        tasks = TaskService(TaskRepository(pool, TaskLimits()), {},
                            managed_plugins=object() if configured else None)
        files = SimpleNamespace(repository=SimpleNamespace(sources=sources)) if files_present else None
        recording = SimpleNamespace(sources=SimpleNamespace(pool=object()) if wrong_pool else sources)
        services = SimpleNamespace(background_task_service=tasks, workspace_notes=object(),
            workspace_presentations=SimpleNamespace(resolver=SimpleNamespace(files=files), exporter=None),
            recording_admission=recording if recording_present else None, recording_policy=None)
        provider = FakeProvider()
        session = LeanRealtimeAudioSession(websocket=FakeSocket(),
            context=RealtimeSessionContext(audio_session_id=str(uuid4()), owner_id="lean-test",
                jarvis_session_id=uuid4(), client_kind="phone_audio", session=SimpleNamespace()),
            settings=JarvisSettings(_env_file=None),
            repository=SimpleNamespace(get_recent_messages=AsyncMock(), create_message=AsyncMock()),
            offer="v=0", provider_connection=provider, runtime_services=services,
            tool_execution_repository=SimpleNamespace() if journal_present else None,
            tool_modules=(), context_providers=())
        self.addAsyncCleanup(session.close, "test_done")
        await session._send_session_update()
        return session, {tool["name"] for tool in provider.sent[-1]["session"]["tools"]}

    async def test_configured_media_is_forwarded_to_real_workspace_dispatcher(self):
        session, names = await self.make()
        self.assertIn("start_media_task", names)
        self.assertTrue(session._lean_workspace.handles("start_media_task"))
        self.assertIn("live_briefing", names)

    async def test_missing_media_dependencies_preserve_ordinary_voice(self):
        for changes in ({"configured": False}, {"files_present": False}, {"recording_present": False},
                        {"wrong_pool": True}, {"journal_present": False}):
            with self.subTest(changes=changes):
                _, names = await self.make(**changes)
                self.assertNotIn("start_media_task", names)
                self.assertIn("live_briefing", names)


if __name__ == "__main__":
    unittest.main()
