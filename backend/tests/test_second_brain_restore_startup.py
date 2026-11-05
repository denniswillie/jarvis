"""Restore startup must fail before any service or paid worker starts."""
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from pydantic import ValidationError

from app.main import lifespan
from app.settings import JarvisSettings


class RestoreStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_restore_barrier_precedes_every_runtime_initializer(self):
        settings = JarvisSettings(_env_file=None, JARVIS_RESTORE_IN_PROGRESS=True)
        with patch("app.main.get_settings", return_value=settings):
            with patch("app.main.JarvisRuntimeServices") as services:
                with self.assertRaisesRegex(RuntimeError, "jarvis_restore_in_progress"):
                    async with lifespan(FastAPI()):
                        self.fail("A restoring application admitted requests")
                services.assert_not_called()

    async def test_normal_startup_remains_available_when_barrier_is_off(self):
        settings = JarvisSettings(_env_file=None, JARVIS_RESTORE_IN_PROGRESS=False)
        with patch("app.main.get_settings", return_value=settings), \
             patch("app.main._initialize_runtime_services", new_callable=AsyncMock) as initialize, \
             patch("app.main.initialize_task_runtime", new_callable=AsyncMock) as tasks, \
             patch("app.main.initialize_knowledge_runtime", new_callable=AsyncMock) as knowledge, \
             patch("app.main.initialize_attention_runtime", new_callable=AsyncMock) as attention, \
             patch("app.main.JarvisRuntimeServices.close", new_callable=AsyncMock) as close:
            async with lifespan(FastAPI()):
                for initializer in (initialize, tasks, knowledge, attention):
                    initializer.assert_awaited_once()
            close.assert_awaited_once()

    def test_restore_flag_is_strictly_parsed_and_defaults_off(self):
        self.assertFalse(JarvisSettings(_env_file=None, JARVIS_RESTORE_IN_PROGRESS=False).JARVIS_RESTORE_IN_PROGRESS)
        with self.assertRaises(ValidationError):
            JarvisSettings(_env_file=None, JARVIS_RESTORE_IN_PROGRESS="unknown")
