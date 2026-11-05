"""Native context/catalog tests without gateway launches, providers, or DB imports."""

from contextvars import ContextVar
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from app.hermes_sessions.bindings import BindingError
from app.hermes_sessions.domain_plugin import capture_native_context, register_domain_tools


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


class DomainPluginTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jarvis-domain-plugin-")
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        (self.home / "jarvis").mkdir()
        self.submission = ContextVar("test_submission", default=None)
        self.call = ContextVar("test_call", default=None)
        self.turn = ContextVar("test_turn", default=None)
        self.worker = Mock(return_value=False)
        self.invoke = Mock(return_value={"content": "fixture"})
        self.modules = {
            "agent.runtime_session_store": module("agent.runtime_session_store", is_worker_process=self.worker),
            "gateway.session_display": module("gateway.session_display", _submission_turn=self.submission),
            "tools.approval_context": module("tools.approval_context",
                                              _approval_tool_call_id=self.call, _approval_turn_id=self.turn),
            "hermes_constants": module("hermes_constants", get_hermes_home=lambda: self.home),
            "app.hermes_sessions.domain_transport": module("app.hermes_sessions.domain_transport",
                                                            invoke_domain_tool=self.invoke),
        }
        self.patcher = patch.dict(sys.modules, self.modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.ctx = Mock()
        self.schema = {"name": "read_fixture", "description": "Read fixture",
                       "parameters": {"type": "object", "properties": {}}}
        self.write_catalog()

    def write_catalog(self, *, enabled=True, tools=None):
        (self.home / "jarvis-adapter.json").write_text(json.dumps({"domain_tools": enabled}))
        (self.home / "jarvis" / "domain-tools.json").write_text(json.dumps(
            {"version": 1, "tools": [self.schema] if tools is None else tools}))

    def context(self, admission="admission-a", input_id="input-a"):
        self.submission.set((admission, input_id))
        self.call.set("tool-call")
        self.turn.set("AGENT-TURN-NOT-INPUT")

    def handler(self):
        register_domain_tools(self.ctx)
        entry = self.ctx.register_tool.call_args.kwargs
        self.assertEqual(entry["toolset"], "jarvis_domain")
        self.assertFalse(entry["is_async"])
        return entry["handler"]

    def test_actual_tuple_not_turn_or_model_metadata_and_queued_inputs_remain_distinct(self):
        handler = self.handler()
        arguments = {"input_id": "FORGED", "admission_id": "FORGED", "session_id": "FORGED"}
        for admission, input_id in (("admission-a", "input-a"), ("admission-b", "input-b")):
            self.context(admission, input_id)
            captured = capture_native_context("native-tip")
            self.assertEqual(captured.turn_id, "AGENT-TURN-NOT-INPUT")
            self.assertEqual(json.loads(handler(arguments, session_id="native-tip")), {"content": "fixture"})
            self.invoke.assert_called_with(self.home.resolve(), native_session_id="native-tip",
                                            input_id=input_id, admission_id=admission,
                                            call_id="tool-call", name="read_fixture", arguments=arguments)

    def test_missing_context_and_managed_subprocess_never_call_transport(self):
        handler = self.handler()
        self.assertIn("error", json.loads(handler({}, session_id="native")))
        for field in ("session", "submission", "call", "turn"):
            self.context()
            if field == "submission":
                self.submission.set(("admission", None))
            if field == "call":
                self.call.set(None)
            if field == "turn":
                self.turn.set(None)
            result = handler({}, session_id=None if field == "session" else "native")
            self.assertIn("error", json.loads(result))
        self.context()
        self.worker.return_value = True
        self.assertIn("managed subprocess", json.loads(handler({}, session_id="native"))["error"])
        self.invoke.assert_not_called()

    def test_image_envelope_reaches_native_registry_as_dictionary(self):
        handler = self.handler()
        self.context()
        envelope = {"_multimodal": True, "content": [], "text_summary": "snapshot"}
        self.invoke.return_value = envelope
        self.assertIs(handler({}, session_id="native-tip"), envelope)

    def test_no_opt_in_means_no_native_domain_tools(self):
        self.write_catalog(enabled=False)
        register_domain_tools(self.ctx)
        self.ctx.register_tool.assert_not_called()

    def test_catalog_is_validated_before_any_registration(self):
        for tools in ([self.schema, self.schema], [{**self.schema, "name": "work_session"}],
                      [self.schema, {"name": "bad"}]):
            self.write_catalog(tools=tools)
            with self.assertRaises(BindingError):
                register_domain_tools(self.ctx)
        self.ctx.register_tool.assert_not_called()

    def test_new_native_imports_do_not_load_backend_db_or_domain_executor(self):
        script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'asyncpg', 'fastapi', 'supabase'} or fullname in {
            'app.hermes_sessions.domain', 'app.hermes_sessions.domain_capture'}:
            raise AssertionError('Forbidden native import: ' + fullname)
sys.meta_path.insert(0, Block())
import app.hermes_sessions.domain_service
import app.hermes_sessions.domain_plugin
import app.hermes_sessions.plugin
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
