"""Command observations distinguish continued work from final application success."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.sandbox.broker import SandboxCommandBroker
from app.sandbox.tools import SandboxToolBridge
from app.sandbox.models import SandboxPolicy


class CommandGuidanceTests(unittest.IsolatedAsyncioTestCase):
    async def observe(self, phase, *, exit_code=None):
        identity = uuid4()
        row = {"phase": phase, "parameters": {"operation_key": "synthetic-ui-input"}, "result": None,
               "allocation_id": uuid4(), "request_digest": "synthetic", "submission_token": uuid4()}
        if phase == "completed":
            row["result"] = {"started": True, "exit_code": exit_code, "timed_out": False,
                "output_limited": False, "runner_error": None, "stdout": "", "stderr": ""}
        broker = object.__new__(SandboxCommandBroker)
        broker.context = SimpleNamespace(claim=object(), task=SimpleNamespace(owner_id="fixture-owner"))
        broker.commands = SimpleNamespace(current=AsyncMock(return_value=row), record=AsyncMock(return_value=row),
            uncertain=AsyncMock(), settle=AsyncMock())
        broker.allocations = SimpleNamespace(current=AsyncMock(return_value={}), handle=lambda _: object())
        broker.provider = SimpleNamespace(command_receipt=AsyncMock(return_value=object() if phase == "running" else None))
        return (await broker.observe(identity))["sandbox_command"], broker

    async def test_running_command_explains_continuation_without_reexecution(self):
        value, broker = await self.observe("running")
        self.assertEqual(value["state"], "running")
        self.assertEqual(value["operation_key"], "synthetic-ui-input")
        self.assertIn("sandbox_status", value["meaning"])
        self.assertIn("Do not pause the whole task", value["meaning"])
        self.assertIn("budget", value["meaning"])
        self.assertNotIn("exit_code", value)
        broker.commands.settle.assert_not_awaited()

    async def test_completed_command_does_not_claim_application_success(self):
        for exit_code in (0, 1):
            with self.subTest(exit_code=exit_code):
                value, broker = await self.observe("completed", exit_code=exit_code)
                self.assertEqual(value["state"], "completed")
                self.assertEqual(value["exit_code"], exit_code)
                self.assertIn("does not prove", value["meaning"])
                broker.provider.command_receipt.assert_not_awaited()
                broker.commands.settle.assert_awaited_once()

    async def test_unknown_receipt_does_not_authorize_new_execution(self):
        value, broker = await self.observe("unknown")
        self.assertEqual(value["state"], "unknown")
        self.assertIn("do not rerun under a new key", value["meaning"])
        self.assertNotIn("exit_code", value)
        broker.commands.uncertain.assert_awaited_once()
        broker.commands.settle.assert_not_awaited()

    def test_desktop_tools_explain_generic_discovery_without_case_specific_programs(self):
        base = SimpleNamespace(context=object(), registry=Mock())
        with patch("app.sandbox.tools.SandboxCommandBroker"), \
                patch("app.sandbox.tools.SandboxInputService") as inputs, \
                patch("app.sandbox.tools.SandboxImageTools"):
            inputs.return_value.definitions.return_value = ()
            bridge = SandboxToolBridge(base, object(), uuid4(), SandboxPolicy(template="desktop"))
        execute = next(tool for tool in bridge.definitions if tool.name == "sandbox_execute")
        self.assertIn("command -v", execute.description)
        self.assertIn("inspect its log", execute.description)
        self.assertNotIn("chromium", execute.description)
