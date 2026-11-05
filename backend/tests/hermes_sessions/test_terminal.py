"""Offline private-socket + real retained services, with a controlled provider."""
import asyncio
from contextvars import ContextVar
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.hermes_sessions.bindings import SessionBindings
from app.hermes_sessions.domain_plugin import NativeInputContext
from app.hermes_sessions.domain_transport import DomainToolServer, invoke_domain_tool
from app.hermes_sessions.sandbox_bindings import SessionSandboxBindings
from app.hermes_sessions.sandboxes import RetainedSandboxService
from app.hermes_sessions.service import RetainedSessionService
from app.hermes_sessions.terminal_adapter import RetainedTerminalAdapter
from app.hermes_sessions.terminal_plugin import RetainedTerminalEnvironment, register_terminal_provider
from app.hermes_sessions.terminal_transport import WORKSPACE, invoke_terminal
from app.sandbox.models import SandboxError, SandboxPolicy
from test_service import NativeFixture
from test_sandboxes import Provider


class Commands(Provider):
    def __init__(self):
        super().__init__()
        self.starts = []
        self.start_error = False
        self.block = False
        self.entered = asyncio.Event()

    async def start(self, handle, command, *, operation_id, cwd):
        self.starts.append((handle, command, operation_id, cwd))
        self.entered.set()
        if self.start_error:
            raise SandboxError("sandbox_command_observation_lost")

        async def wait(*, timeout_seconds):
            if self.block:
                await asyncio.Event().wait()
            return SimpleNamespace(stdout="fixture output", stderr="", exit_code=0)
        return SimpleNamespace(wait=wait)


class TerminalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jarvis-terminal-")
        self.home = Path(self.temp.name).resolve()
        self.bindings = SessionBindings(self.home / "bindings.sqlite", owner_id="owner", profile_home=self.home)
        self.native = NativeFixture(self.home)
        self.sessions = RetainedSessionService(self.native, self.bindings,
            launch={"source": "cli", "cwd": str(self.home), "toolsets": []})
        self.master = await self.sessions.master()
        self.child = await self.sessions.create_work(self.master.logical_id, request_id="child", label="Child")
        self.store = SessionSandboxBindings(self.home / "resources.sqlite", owner_id="owner", profile_id="fixture")
        self.provider = Commands()
        self.sandboxes = RetainedSandboxService(self.store, self.provider, policy=SandboxPolicy())
        self.adapter = RetainedTerminalAdapter(sessions=self.sessions, sandboxes=self.sandboxes)
        self.server = await DomainToolServer(self.home, None, terminal=self.adapter).start()
        self.context = NativeInputContext(self.master.native_id, "input-a", "admission-a", "call-a", "TURN-not-input")

    async def asyncTearDown(self):
        await self.server.close()
        self.store.close()
        self.temp.cleanup()

    async def rpc(self, operation, *, context=None, attachment=None, arguments=None):
        return await asyncio.to_thread(invoke_terminal, self.home, operation=operation,
            context=context, attachment=attachment, arguments=arguments)

    async def attach(self, context=None):
        return (await self.rpc("attach", context=context or self.context))["attachment"]

    async def execute(self, token, *, context=None, **changes):
        args = {"command": "printf fixture", "cwd": WORKSPACE, "timeout": 30} | changes
        return await self.rpc("execute", context=context or self.context, attachment=token, arguments=args)

    async def test_attach_is_lazy_and_two_inputs_reuse_one_resource(self):
        token = await self.attach()
        self.assertEqual(self.provider.creates, 0)
        self.assertEqual((await self.execute(token))["status"], "completed")
        # Same tool call ID and turn, different admission/input: not conflated.
        next_input = replace(self.context, input_id="input-b", admission_id="admission-b")
        self.assertEqual((await self.execute(token, context=next_input))["status"], "completed")
        self.assertEqual(self.provider.creates, 1)
        self.assertNotEqual(self.provider.starts[0][2], self.provider.starts[1][2])
        self.assertEqual(self.provider.starts[0][3], ".")

    async def test_admission_is_preserved_independently_of_input_and_turn(self):
        token = await self.attach()
        await self.execute(token)
        await self.execute(token, context=replace(self.context, admission_id="different-admission"))
        self.assertEqual(len(self.provider.starts), 2)
        self.assertNotEqual(self.provider.starts[0][2], self.provider.starts[1][2])

    async def test_cross_session_attachment_and_foreign_native_are_refused(self):
        token = await self.attach()
        other = replace(self.context, native_session_id=self.child.native_id)
        self.assertEqual((await self.execute(token, context=other))["error"]["code"], "terminal_attachment_mismatch")
        self.assertEqual((await self.execute(token, context=replace(other, native_session_id="foreign")))["status"], "refused")
        self.assertEqual(self.provider.starts, [])

    async def test_compaction_tip_resolves_to_same_logical_attachment(self):
        token = await self.attach()
        snapshot = self.native.snapshot
        async def compacted(native_id):
            return await snapshot(self.master.native_id if native_id == "compacted-tip" else native_id)
        self.native.snapshot = compacted
        result = await self.execute(token, context=replace(self.context, native_session_id="compacted-tip"))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.provider.starts[0][0].scope.logical_session_id, self.master.logical_id)

    async def test_no_model_identity_or_background_fields_and_workspace_escape(self):
        token = await self.attach()
        for changes in ({"session_id": self.child.native_id}, {"background": True}, {"cwd": "/tmp"},
                        {"cwd": "../other"}, {"timeout": True}, {"timeout": 301}):
            self.assertNotEqual((await self.execute(token, **changes))["status"], "completed")
        self.assertEqual(self.provider.starts, [])

    async def test_unknown_start_and_repeated_call_never_resubmit(self):
        token = await self.attach()
        self.provider.start_error = True
        self.assertEqual((await self.execute(token))["status"], "unknown")
        self.assertEqual((await self.execute(token))["error"]["code"], "terminal_call_already_attempted")
        self.assertEqual(len(self.provider.starts), 1)

    async def test_transport_timeout_does_not_cancel_claim_or_retry_command(self):
        self.server.timeout = .1
        self.provider.block = True
        token = await self.attach()
        result = await self.execute(token)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual((await self.execute(token))["error"]["code"], "terminal_call_already_attempted")
        self.assertEqual(len(self.provider.starts), 1)

    async def test_cleanup_only_detaches_and_paused_resource_never_auto_resumes(self):
        token = await self.attach()
        await self.execute(token)
        self.assertEqual((await self.rpc("detach", attachment=token))["status"], "detached")
        self.assertEqual((self.provider.pauses, self.provider.resumes), (0, 0))
        await self.sandboxes.pause(self.master.logical_id)
        token = await self.attach()
        result = await self.execute(token, context=replace(self.context, call_id="next-call"))
        self.assertEqual(result["error"]["code"], "session_sandbox_paused")
        self.assertEqual(self.provider.resumes, 0)
        self.assertEqual(len(self.provider.starts), 1)

    async def test_server_restart_invalidates_attachment_without_allocating(self):
        token = await self.attach()
        await self.server.close()
        self.adapter = RetainedTerminalAdapter(sessions=self.sessions, sandboxes=self.sandboxes)
        self.server = await DomainToolServer(self.home, None, terminal=self.adapter).start()
        self.assertEqual((await self.execute(token))["error"]["code"], "terminal_attachment_expired")
        self.assertEqual(self.provider.creates, 0)

    async def test_native_proxy_reattaches_after_backend_restart(self):
        env = RetainedTerminalEnvironment(self.home)
        with patch("app.hermes_sessions.terminal_plugin._current_context", return_value=self.context):
            first = await asyncio.to_thread(env.execute, "printf first", bounded_capture=True)
            self.assertEqual(first["returncode"], 0)
        await self.server.close()
        self.adapter = RetainedTerminalAdapter(sessions=self.sessions, sandboxes=self.sandboxes)
        self.server = await DomainToolServer(self.home, None, terminal=self.adapter).start()
        next_input = replace(self.context, input_id="next", admission_id="next", call_id="next")
        with patch("app.hermes_sessions.terminal_plugin._current_context", return_value=next_input):
            second = await asyncio.to_thread(env.execute, "printf second", bounded_capture=True)
            self.assertEqual(second["returncode"], 0)
        self.assertEqual(self.provider.creates, 1)
        self.assertEqual(len(self.provider.starts), 2)
        await asyncio.to_thread(env.cleanup)

    async def test_original_domain_envelope_and_terminal_only_composition(self):
        args = dict(native_session_id="native", input_id="input", admission_id="admission",
                    call_id="call", name="read", arguments={})
        result = await asyncio.to_thread(invoke_domain_tool, self.home, **args)
        self.assertEqual(result["error"]["code"], "domain_disabled")
        domain = SimpleNamespace(invoke=AsyncMock(return_value=SimpleNamespace(payload={"read": True}, image=None)))
        self.server.adapter = domain
        self.assertEqual(await asyncio.to_thread(invoke_domain_tool, self.home, **args), {"read": True})
        self.assertEqual((await self.execute(await self.attach()))["status"], "completed")

    async def test_native_proxy_captures_fresh_context_and_no_identity_from_task_id(self):
        session, submission, call, turn = [ContextVar(name, default=None) for name in ("session", "submission", "call", "turn")]
        modules = {}
        for name, fields in {
            "tools.approval_context": dict(_approval_session_id=session, _approval_tool_call_id=call, _approval_turn_id=turn),
            "gateway.session_display": dict(_submission_turn=submission),
            "agent.runtime_session_store": dict(is_worker_process=lambda: False),
        }.items():
            mod = ModuleType(name)
            mod.__dict__.update(fields)
            modules[name] = mod
        env = RetainedTerminalEnvironment(self.home)
        async def run():
            return await asyncio.to_thread(env.execute, "printf fixture", bounded_capture=True, timeout=30)
        with patch.dict(sys.modules, modules):
            self.assertEqual((await run())["returncode"], 125)
            session.set(self.master.native_id)
            submission.set(("admission-a", "input-a"))
            call.set("call")
            turn.set("not-input")
            self.assertEqual((await run())["returncode"], 0)
            submission.set(("admission-b", "input-b"))
            self.assertEqual((await run())["returncode"], 0)
            session.set(self.child.native_id)
            self.assertIn("attachment_mismatch", (await run())["output"])
            modules["agent.runtime_session_store"].is_worker_process = lambda: True
            self.assertEqual((await run())["returncode"], 125)
        await asyncio.to_thread(env.cleanup)
        self.assertEqual(len(self.provider.starts), 2)
        self.assertEqual(self.adapter._attachments, {})

    async def test_native_provider_and_real_foreground_dispatch_over_socket(self):
        # Separate process prevents pinned native module globals affecting other tests.
        source = Path(__file__).resolve().parents[3] / ".jarvis-dev/hermes-session-h0.Ld4B6Z/source"
        (self.home / "jarvis-adapter.json").write_text(json.dumps({"retained_terminal": True}))
        script = r'''
import json, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from app.hermes_sessions.terminal_plugin import register_terminal_provider
from agent.terminal_env_registry import register_provider
from tools.terminal_tool_backends import _build_plugin_env
from tools.approval_context import _approval_session_id, _approval_tool_call_id, _approval_turn_id
from gateway.session_display import _submission_turn
from tools import terminal_tool as tt
_approval_session_id.set(sys.argv[1]); _approval_tool_call_id.set("native-call")
_approval_turn_id.set("native-turn-not-input"); _submission_turn.set(("native-admission", "native-input"))
register_terminal_provider(SimpleNamespace(register_terminal_environment_provider=register_provider))
env = _build_plugin_env(env_type="jarvis_e2b", image=None, cwd="/ignored", timeout=30, cc={}, task_id="FORGED-CACHE-KEY")
plan = SimpleNamespace(env_type="jarvis_e2b", effective_task_id="FORGED-CACHE-KEY", effective_timeout=30,
                       cwd="/home/jarvis/workspace", host_cwd=None)
with patch.object(tt, "_resolve_command_cwd", return_value="/home/jarvis/workspace"):
    result = tt._run_foreground("printf fixture", env, plan, task_id="FORGED-CACHE-KEY",
        session_id=sys.argv[1], session_key="fixture", workdir=None, approval_note=None,
        clear_interrupt=False, metered=False)
env.cleanup()
print("RESULT=" + result)
'''
        environment = dict(os.environ, HERMES_HOME=str(self.home), PYTHONPATH=str(source) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        process = await asyncio.create_subprocess_exec(sys.executable, "-c", script, self.master.native_id,
            env=environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(process.communicate(), 25)
        self.assertEqual(process.returncode, 0, stderr.decode())
        line = next(line for line in stdout.decode().splitlines() if line.startswith("RESULT="))
        self.assertEqual(json.loads(line[7:])["exit_code"], 0)
        self.assertEqual(len(self.provider.starts), 1)
        self.assertEqual(self.provider.starts[0][0].scope.logical_session_id, self.master.logical_id)


class TerminalPluginTests(unittest.TestCase):
    def test_unknown_execution_is_not_retried(self):
        env = RetainedTerminalEnvironment("/tmp")
        with patch("app.hermes_sessions.terminal_plugin._current_context", return_value=object()), \
                patch("app.hermes_sessions.terminal_plugin.invoke_terminal", side_effect=[
                    {"status": "attached", "attachment": "first"},
                    {"status": "unknown", "error": {"code": "terminal_execution_uncertain"}},
                ]) as rpc:
            result = env.execute("printf fixture", bounded_capture=True)
        self.assertEqual(result["returncode"], 125)
        self.assertEqual(rpc.call_count, 2)

    def test_reconnection_is_bounded_to_one_attempt(self):
        env = RetainedTerminalEnvironment("/tmp")
        expired = {"status": "refused", "error": {"code": "terminal_attachment_expired"}}
        with patch("app.hermes_sessions.terminal_plugin._current_context", return_value=object()), \
                patch("app.hermes_sessions.terminal_plugin.invoke_terminal", side_effect=[
                    {"status": "attached", "attachment": "first"}, expired,
                    {"status": "attached", "attachment": "second"}, expired,
                ]) as rpc:
            result = env.execute("printf fixture", bounded_capture=True)
        self.assertEqual(result["returncode"], 125)
        self.assertEqual(rpc.call_count, 4)

    def test_configured_provider_visible_before_backend_socket_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "jarvis-adapter.json").write_text('{"retained_terminal": true}')
            constants = ModuleType("hermes_constants")
            constants.get_hermes_home = lambda: home
            provider_module = ModuleType("agent.terminal_env_provider")
            provider_module.TerminalEnvironmentProvider = object
            ctx = Mock()
            with patch.dict(sys.modules, {"hermes_constants": constants,
                                         "agent.terminal_env_provider": provider_module}):
                register_terminal_provider(ctx)
            provider = ctx.register_terminal_environment_provider.call_args.args[0]
            self.assertTrue(provider.is_available())
            self.assertIn("xdotool type --delay 0", provider.env_description)
            self.assertIn("stopping as soon as ready", provider.env_description)
            self.assertIn("Inspect the screenshot", provider.env_description)
            env = provider.create_environment(cwd="/ignored", timeout=30)
            with patch("app.hermes_sessions.terminal_plugin._current_context", return_value=None):
                result = env.execute("printf fixture", bounded_capture=True)
            self.assertEqual(result["returncode"], 125)
            self.assertIn("No complete terminal result", result["output"])

    def test_opt_in_and_no_native_db_provider_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            constants = ModuleType("hermes_constants")
            constants.get_hermes_home = lambda: home
            ctx = Mock()
            with patch.dict(sys.modules, {"hermes_constants": constants}):
                register_terminal_provider(ctx)
                (home / "jarvis-adapter.json").write_text('{"retained_terminal": false}')
                register_terminal_provider(ctx)
            ctx.register_terminal_environment_provider.assert_not_called()

    def test_background_and_internal_calls_never_touch_transport(self):
        env = RetainedTerminalEnvironment("/tmp")
        with patch("app.hermes_sessions.terminal_plugin.invoke_terminal") as rpc:
            for args in ({}, {"bounded_capture": True, "rewrite_compound_background": False},
                         {"bounded_capture": True, "stdin_data": "x"},
                         {"bounded_capture": True, "yield_handler": object()}):
                self.assertEqual(env.execute("printf fixture", **args)["returncode"], 125)
            with self.assertRaisesRegex(RuntimeError, "not qualified"):
                env.get_temp_dir()
            env.cleanup()
            rpc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
