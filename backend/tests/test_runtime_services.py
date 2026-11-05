from __future__ import annotations

import asyncio
import unittest
from typing import cast

from app.files.storage import JarvisFileStorageClient
from app.memory.embeddings import ExperienceEventEmbeddingService
from app.realtime.session_manager import RealtimeSessionManager
from app.reminders.delivery import ReminderDeliveryService
from app.runtime.concurrency import PriorityAsyncLimiter
from app.runtime.events import DomainEventBus
from app.runtime.services import JarvisRuntimeServices
from app.perception.frame_window import RecentVisualFrameWindow
from app.settings import JarvisSettings
from app.visual_memory.latest_frame_buffer import LatestVisualFrameBuffer
from app.visual_memory.services import VisualMemoryServices
from app.visual_memory.storage import VisualFrameStorageClient


class _Closable:
    def __init__(self, label: str, events: list[str]) -> None:
        self._label = label
        self._events = events

    async def close(self) -> None:
        self._events.append(self._label)


class _RealtimeManager:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def close_all(self) -> None:
        self._events.append("realtime")


class RuntimeServicesTests(unittest.IsolatedAsyncioTestCase):
    async def test_canonical_capture_closes_before_storage_without_a_memory_provider(self) -> None:
        events = []
        services = JarvisRuntimeServices(
            settings=cast(JarvisSettings, object()),
            realtime_session_manager=cast(RealtimeSessionManager, _RealtimeManager(events)),
            sensory_frame_window=cast(RecentVisualFrameWindow, object()),
            latest_visual_frame_buffer=cast(LatestVisualFrameBuffer, object()),
            domain_event_bus=DomainEventBus(),
            model_limiter=PriorityAsyncLimiter(1),
            db_pool=cast(object, _Closable("pool", events)),
            visual_frame_storage=cast(VisualFrameStorageClient, _Closable("visual-storage", events)),
            background_task_context_service=cast(object, object()),
        )
        await services.close()
        await services.close()
        self.assertEqual(events, ["realtime", "visual-storage", "pool"])

    async def test_visual_and_experience_producers_close_before_knowledge(self):
        events = []
        services = JarvisRuntimeServices(
            settings=cast(JarvisSettings, object()),
            realtime_session_manager=cast(RealtimeSessionManager, _RealtimeManager(events)),
            sensory_frame_window=cast(RecentVisualFrameWindow, object()),
            latest_visual_frame_buffer=cast(LatestVisualFrameBuffer, object()),
            domain_event_bus=DomainEventBus(), model_limiter=PriorityAsyncLimiter(1),
            db_pool=cast(object, _Closable("pool", events)),
            knowledge_runtime=cast(object, _Closable("knowledge", events)),
            visual_memory_services=cast(VisualMemoryServices, _Closable("visual", events)),
            experience_event_service=cast(object, _Closable("experience", events)),
        )
        await services.close()
        await services.close()
        self.assertEqual(events, ["realtime", "visual", "experience", "knowledge", "pool"])

    async def test_close_is_reverse_order_and_idempotent(self) -> None:
        events: list[str] = []

        async def background(label: str) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                events.append(label)

        retention_task = asyncio.create_task(background("retention"))
        reminder_task = asyncio.create_task(background("scheduler"))
        await asyncio.sleep(0)

        services = JarvisRuntimeServices(
            settings=cast(JarvisSettings, object()),
            realtime_session_manager=cast(
                RealtimeSessionManager,
                _RealtimeManager(events),
            ),
            sensory_frame_window=cast(RecentVisualFrameWindow, object()),
            latest_visual_frame_buffer=cast(LatestVisualFrameBuffer, object()),
            domain_event_bus=DomainEventBus(),
            model_limiter=PriorityAsyncLimiter(1),
            db_pool=cast(object, _Closable("pool", events)),
            visual_frame_storage=cast(
                VisualFrameStorageClient,
                _Closable("visual-storage", events),
            ),
            jarvis_file_storage=cast(
                JarvisFileStorageClient,
                _Closable("file-storage", events),
            ),
            visual_memory_services=cast(
                VisualMemoryServices,
                _Closable("visual-memory", events),
            ),
            experience_event_embedding_service=cast(
                ExperienceEventEmbeddingService,
                _Closable("event-embeddings", events),
            ),
            reminder_delivery_service=cast(
                ReminderDeliveryService,
                _Closable("reminder-delivery", events),
            ),
            visual_frame_retention_task=retention_task,
            reminder_scheduler_task=reminder_task,
        )

        await services.close()
        await services.close()

        self.assertEqual(
            events,
            [
                "scheduler",
                "reminder-delivery",
                "retention",
                "visual-memory",
                "event-embeddings",
                "file-storage",
                "visual-storage",
                "pool",
                "realtime",
            ],
        )


if __name__ == "__main__":
    unittest.main()
