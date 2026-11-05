"""Offline opt-in composition and resource-order checks, without model calls."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.shared_memory.client import BankScope
from app.shared_memory.configuration import initialize_shared_memory, close_shared_memory
from app.runtime.services import JarvisRuntimeServices


class FakeIntake:
    def __init__(self, memory, source, path, *, invalidate):
        self.memory, self.source, self.path, self.invalidate = memory, source, path, invalidate
        self.initialize = AsyncMock()
        self.close = AsyncMock()
        self.abort = Mock()
        self.issue = None

    async def run(self, stop):
        await stop.wait()


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def services(self):
        return SimpleNamespace(settings=SimpleNamespace(JARVIS_OWNER_ID="owner"), db_pool=object(),
                               shared_memory=None, shared_memory_intake=None,
                               shared_memory_task=None, shared_memory_stop=None,
                               shared_memory_invalidator=None, shared_memory_issue=None)

    async def test_disabled_flag_does_not_construct_source_or_worker(self):
        services = self.services()
        with patch("app.shared_memory.configuration.AsyncpgCaptionSource") as source:
            await initialize_shared_memory(services, environ={})
        self.assertIsNone(services.shared_memory)
        self.assertIsNone(services.shared_memory_task)
        source.assert_not_called()

    async def test_owner_bound_initialization_and_dynamic_honest_invalidator(self):
        services = self.services()
        memory = SimpleNamespace(scope=BankScope.for_owner("owner", generation="test"), aclose=AsyncMock())
        with patch("app.shared_memory.configuration.configured_memory", return_value=memory), \
             patch("app.shared_memory.configuration.LiveDialogueIntake", FakeIntake):
            await initialize_shared_memory(services, environ={"JARVIS_NATIVE_MEMORY_CHECKPOINT_FILE": "/private/test.sqlite"})
        intake = services.shared_memory_intake
        self.assertEqual(intake.source.owner_id, "owner")
        self.assertEqual(intake.path, Path("/private/test.sqlite"))
        intake.initialize.assert_awaited_once()
        with self.assertRaisesRegex(RuntimeError, "invalidator_not_configured"):
            await intake.invalidate(object())
        services.shared_memory_invalidator = AsyncMock()
        scope = object()
        await intake.invalidate(scope)
        services.shared_memory_invalidator.assert_awaited_once_with(scope)
        await close_shared_memory(services)
        intake.close.assert_awaited_once()
        intake.abort.assert_called_once()
        memory.aclose.assert_awaited_once()
        self.assertIsNone(services.shared_memory_task)

    async def test_shutdown_failure_preserves_pending_issue_and_closes_client(self):
        services = self.services()
        services.shared_memory = SimpleNamespace(aclose=AsyncMock())
        memory = services.shared_memory
        services.shared_memory_intake = FakeIntake(memory, None, Path("/tmp/test"), invalidate=AsyncMock())
        intake = services.shared_memory_intake
        intake.close.side_effect = RuntimeError("source unavailable")
        await close_shared_memory(services)
        self.assertEqual(services.shared_memory_issue, "native_memory_flush_pending")
        intake.abort.assert_called_once()
        memory.aclose.assert_awaited_once()

    async def test_runtime_closes_capture_then_memory_then_database(self):
        order = []
        async def record(label):
            order.append(label)
        services = JarvisRuntimeServices(
            settings=SimpleNamespace(),
            realtime_session_manager=SimpleNamespace(close_all=lambda: record("capture")),
            sensory_frame_window=None, latest_visual_frame_buffer=None,
            domain_event_bus=None, model_limiter=None,
        )
        services.db_pool = SimpleNamespace(close=lambda: record("database"))
        services.shared_memory = SimpleNamespace(aclose=lambda: record("memory-client"))
        services.shared_memory_intake = SimpleNamespace(close=lambda: record("intake"), abort=lambda: None)
        await services.close()
        await services.close()
        self.assertEqual(order, ["capture", "intake", "memory-client", "database"])
