"""The interactive file fixture must unwind its owners before process exit."""
import signal
import unittest
from unittest.mock import patch

from fastapi import FastAPI
import uvicorn

from evals.second_brain.file_phone_fixture import FixtureServer


class FilePhoneFixtureShutdownTests(unittest.TestCase):
    def test_shutdown_signal_returns_to_outer_resource_cleanup(self):
        for shutdown_signal in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=shutdown_signal):
                server = FixtureServer(uvicorn.Config(FastAPI(), log_config=None))
                with patch("uvicorn.server.signal.raise_signal") as reraise:
                    with server.capture_signals():
                        server.handle_exit(shutdown_signal, None)
                    self.assertTrue(server.should_exit)
                    reraise.assert_not_called()
