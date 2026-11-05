from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi import FastAPI

from app.files.storage import JarvisFileStorageClient
from app.main import _create_tool_modules, lifespan
from app.perception.current_image import CurrentImageService
from app.perception.tools import CurrentImageToolModule
from app.runtime.events import DomainEvent, DomainEventName, VisualFrameAcceptedPayload
from app.settings import JarvisSettings
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry
from app.visual_memory.tools import VisualCaptureToolModule


class _FakePool:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class ApplicationLifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifespan_publishes_runtime_and_composes_cutover_tools(self) -> None:
        settings = JarvisSettings(
            _env_file=None,
            APP_ENV="test",
            JARVIS_STATIC_BEARER_TOKEN="test-token",
            SUPABASE_DB_URL=None,
            OPENAI_API_KEY=None,
            GEMINI_API_KEY=None,
            VISUAL_REALTIME_ENABLED=False,
            VISUAL_FAST_PERCEPTION_ENABLED=False,
        )
        pool = _FakePool()

        async def initialize(services) -> None:
            services.db_pool = pool
            services.jarvis_file_storage = Mock(spec=JarvisFileStorageClient)
            services.tool_modules = _create_tool_modules(services)

        app = FastAPI()
        with (
            patch("app.main.get_settings", return_value=settings),
            patch("app.main._initialize_runtime_services", side_effect=initialize),
        ):
            async with lifespan(app):
                runtime = app.state.runtime_services
                self.assertIs(runtime.db_pool, pool)
                self.assertIsInstance(runtime.current_image_service, CurrentImageService)
                self.assertEqual(runtime.current_image_service._max_age_seconds, settings.VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS)
                self.assertTrue(any(isinstance(module, CurrentImageToolModule) for module in runtime.tool_modules))
                registry = create_tool_registry(modules=runtime.tool_modules, context=ToolBuildContext(
                    settings=settings, owner_id="owner", session_id=uuid4(),
                ))
                self.assertIsNotNone(registry.get("get_current_camera_view"))
                self.assertIsNone(registry.get("get_current_visual_context"))
                self.assertIsNotNone(registry.get("capture_latest_visual_frame"))
                self.assertTrue(
                    any(
                        isinstance(module, VisualCaptureToolModule)
                        for module in runtime.tool_modules
                    )
                )

                session_id = uuid4()
                notify = Mock()
                await runtime.realtime_session_manager.register(
                    audio_session_id="audio", owner_id="owner", jarvis_session_id=session_id,
                    session=SimpleNamespace(notify_visual_frame_available=notify),
                )
                try:
                    dispatch = await runtime.domain_event_bus.publish(DomainEvent(
                        name=DomainEventName.VISUAL_FRAME_ACCEPTED,
                        payload=VisualFrameAcceptedPayload(
                            owner_id="owner", session_id=session_id,
                            frame_id=uuid4(), frame_sequence=1,
                        ),
                    ))
                    self.assertEqual(dispatch.delivered_count, 1)
                    self.assertEqual(dispatch.failed_count, 0)
                    notify.assert_called_once()
                finally:
                    await runtime.realtime_session_manager.unregister(
                        audio_session_id="audio", owner_id="owner", jarvis_session_id=session_id,
                    )

        self.assertTrue(pool.closed)
