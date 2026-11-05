"""Native fixture startup must tolerate a server that has not bound its port yet."""
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from evals.second_brain.computer_use_probe import SERVER_READY_CHECK, trusted_fixture


class ComputerFixtureReadinessTests(unittest.IsolatedAsyncioTestCase):
    def provider(self, states):
        replies = [SimpleNamespace(stdout="", exit_code=0) for _ in range(2)]
        replies += [SimpleNamespace(stdout=state + "\n", exit_code=0) for state in states]
        return SimpleNamespace(_connect=AsyncMock(return_value=SimpleNamespace(commands=SimpleNamespace(run=object()))),
                               _request=AsyncMock(side_effect=replies))

    async def test_waiting_is_not_mistaken_for_ready_or_a_provider_failure(self):
        provider = self.provider(["waiting", "ready"])
        with patch("evals.second_brain.computer_use_probe.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await trusted_fixture(provider, object(), "VIEW-SYNTHETIC")
        self.assertEqual(provider._request.await_count, 4)
        sleep.assert_awaited_once_with(0.1)

    async def test_readiness_exhaustion_is_an_explicit_fixture_error(self):
        provider = self.provider(["waiting"] * 20)
        with patch("evals.second_brain.computer_use_probe.asyncio.sleep", new_callable=AsyncMock), \
                self.assertRaisesRegex(RuntimeError, "computer_fixture_unavailable"):
            await trusted_fixture(provider, object(), "VIEW-SYNTHETIC")
        self.assertEqual(provider._request.await_count, 22)

    def test_remote_readiness_script_reports_a_closed_port_without_nonzero_exit(self):
        output = StringIO()
        with patch("socket.create_connection", side_effect=ConnectionRefusedError), redirect_stdout(output):
            exec(SERVER_READY_CHECK, {})
        self.assertEqual(output.getvalue(), "waiting\n")

    def test_remote_readiness_script_closes_a_successful_socket(self):
        connection = SimpleNamespace(close=Mock())
        output = StringIO()
        with patch("socket.create_connection", return_value=connection), redirect_stdout(output):
            exec(SERVER_READY_CHECK, {})
        self.assertEqual(output.getvalue(), "ready\n")
        connection.close.assert_called_once_with()
